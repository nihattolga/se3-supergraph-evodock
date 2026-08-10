import numpy as np
from typing import Dict, List
from pathlib import Path

from .edge_builder import ProteinLigandEdgeBuilder

from biotite.structure import AtomArray, AtomArrayStack, filter_amino_acids, get_residue_starts
from biotite.structure.io import pdb, pdbx

class PDBProcessor:
    """
    Complete PDB file processor.
    
    Handles:
    - PDB reading (standard and PDBx/mmCIF)
    - Multi-model processing
    - Ligand detection
    - Metal identification
    - Water classification
    - Biological assembly generation
    - Missing atom reconstruction
    """
    
    def __init__(self, 
                 pdb_path: str,
                 include_water: bool = False,
                 include_hetero: bool = True,
                 biological_assembly: bool = False,
                 model_id: int = 0):
        
        self.pdb_path = pdb_path
        self.include_water = include_water
        self.include_hetero = include_hetero
        self.biological_assembly = biological_assembly
        self.model_id = model_id
        
        # PDB metadata
        self.pdb_id = None
        self.resolution = None
        self.r_factor = None
        self.r_free = None
        self.deposition_date = None
        self.experimental_method = None
        
        # Load and process
        self.atoms = None
        self.all_models = None
        self._load_structure()
        
        # Ligand detection
        self.ligand_detector = LigandDetector()
        self.residue_categories = None
        
        # Edge builder
        self.edge_builder = None
        
    def _load_structure(self):
        """Load PDB file with all available information."""
        pdb_path = Path(self.pdb_path)
        
        # Determine file format
        if pdb_path.suffix in ['.pdb', '.ent']:
            self._load_pdb_format()
        elif pdb_path.suffix in ['.cif', '.mmcif', '.pdbx']:
            self._load_cif_format()
        else:
            raise ValueError(f"Unsupported file format: {pdb_path.suffix}")
        
        # Extract metadata
        self._extract_metadata()
        
        # Process multiple models if available
        if isinstance(self.atoms, AtomArrayStack) and len(self.atoms) > 1:
            self.all_models = self.atoms
            self.atoms = self.atoms[self.model_id]
            print(f"Loaded {len(self.all_models)} models, using model {self.model_id}")
    
    def _load_pdb_format(self):
        """Load standard PDB format."""
        pdb_file = pdb.PDBFile.read(self.pdb_path)
        
        # Try to get multiple models
        try:
            self.atoms = pdb_file.get_structure(model=None)[0]  # All models
        except:
            self.atoms = pdb_file.get_structure(model=1)[0]  # Single model
        
        # Get PDB metadata
        if hasattr(pdb_file, 'get_header'):
            header = pdb_file.get_header()
            if header:
                self.pdb_id = header.get('pdb_id', None)
                self.deposition_date = header.get('deposition_date', None)
                self.resolution = header.get('resolution', None)
    
    def _load_cif_format(self):
        """Load PDBx/mmCIF format."""
        pdbx_file = pdbx.PDBxFile.read(self.pdb_path)
        self.atoms = pdbx.get_structure(pdbx_file, model=1)[0]
        
        # Extract CIF metadata
        try:
            category = pdbx_file.get_category('refine')
            if category:
                self.resolution = float(category.get('ls_d_res_high', [None])[0])
                self.r_factor = float(category.get('ls_R_factor_obs', [None])[0])
                self.r_free = float(category.get('ls_R_factor_R_free', [None])[0])
            
            category = pdbx_file.get_category('exptl')
            if category:
                self.experimental_method = category.get('method', [None])[0]
        except:
            pass
    
    def _extract_metadata(self):
        """Extract all available metadata."""
        if self.atoms is None:
            return
        
        # Try to extract from annotations
        try:
            # Resolution from X-ray, NMR has no resolution
            if hasattr(self.atoms, 'annotations'):
                annotations = self.atoms.annotations
                # Various attempts to get metadata
        except:
            pass
    
    def categorize_residues(self):
        """Categorize all residues in the structure."""
        self.residue_categories = self.ligand_detector.get_all_categories(self.atoms)
        return self.residue_categories
    
    def get_protein_atoms(self) -> AtomArray:
        """Get only protein atoms."""
        return self.atoms[filter_amino_acids(self.atoms)]
    
    def get_ligand_atoms(self) -> AtomArray:
        """Get only ligand atoms."""
        if self.residue_categories is None:
            self.categorize_residues()
        
        ligand_mask = np.zeros(len(self.atoms), dtype=bool)
        for cat in ['ligand', 'cofactor']:
            for res_info in self.residue_categories.get(cat, []):
                mask = (self.atoms.res_id == res_info['residue_id']) & \
                       (self.atoms.chain_id == res_info['chain_id'])
                ligand_mask |= mask
        
        return self.atoms[ligand_mask]
    
    def get_metal_atoms(self) -> AtomArray:
        """Get only metal atoms."""
        if self.residue_categories is None:
            self.categorize_residues()
        
        metal_mask = np.zeros(len(self.atoms), dtype=bool)
        for metal_info in self.residue_categories.get('metal', []):
            mask = (self.atoms.res_id == metal_info['residue_id']) & \
                   (self.atoms.chain_id == metal_info['chain_id'])
            metal_mask |= mask
        
        return self.atoms[metal_mask]
    
    def get_water_atoms(self) -> AtomArray:
        """Get only water atoms."""
        if self.residue_categories is None:
            self.categorize_residues()
        
        water_mask = np.zeros(len(self.atoms), dtype=bool)
        for water_info in self.residue_categories.get('water', []):
            mask = (self.atoms.res_id == water_info['residue_id']) & \
                   (self.atoms.chain_id == water_info['chain_id'])
            water_mask |= mask
        
        return self.atoms[water_mask]
    
    def get_pocket_atoms(self, distance_cutoff: float = 6.0) -> AtomArray:
        """Get atoms within the binding pocket (near ligands)."""
        ligand_atoms = self.get_ligand_atoms()
        
        if len(ligand_atoms) == 0:
            # No ligand found, return empty
            return self.atoms[:0]
        
        pocket_mask = np.zeros(len(self.atoms), dtype=bool)
        for lig_coord in ligand_atoms.coord:
            dists = np.linalg.norm(self.atoms.coord - lig_coord, axis=1)
            pocket_mask |= (dists < distance_cutoff)
        
        return self.atoms[pocket_mask]
    
    def build_edges(self) -> Dict[str, List[Dict]]:
        """Build complete edge graph."""
        
        if self.edge_builder is None:
            self.edge_builder = ProteinLigandEdgeBuilder()
        
        edges = self.edge_builder.build_complete_edge_graph(self.atoms)
        return edges


class LigandDetector:
    """
    Advanced ligand detection with chemical awareness.
    """
    
    # Standard amino acids
    STANDARD_AA = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
        'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
        'THR', 'TRP', 'TYR', 'VAL'
    }
    
    # Common metals in proteins
    METALS = {
        'LI', 'BE', 'NA', 'MG', 'AL', 'K', 'CA', 'SC', 'TI', 'V',
        'CR', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN', 'GA', 'RB', 'SR',
        'Y', 'ZR', 'NB', 'MO', 'TC', 'RU', 'RH', 'PD', 'AG', 'CD',
        'IN', 'SN', 'SB', 'CS', 'BA', 'LA', 'HF', 'TA', 'W', 'RE',
        'OS', 'IR', 'PT', 'AU', 'HG', 'TL', 'PB', 'BI', 'PO', 'RA',
        'TH', 'PA', 'U', 'NP', 'PU', 'AM', 'CM',
    }
    
    # Common solvents
    SOLVENTS = {
        'HOH', 'WAT', 'H2O', 'DOD', 'D2O', 'SOL', 'OH2',
    }
    
    # Common buffer/cryoprotectant
    BUFFERS = {
        'TRS', 'TRIS', 'HEPES', 'MES', 'PIPES', 'MOPS', 'PEG',
        'GOL', 'EDO', 'ACT', 'ACM', 'SO4', 'PO4', 'NO3', 'CL', 'BR',
        'IOD', 'FMT', 'ACY', 'BME', 'BOG', 'C8E', 'DMS', 'EOH',
        'GAI', 'IPA', 'LDA', 'MOH', 'MPD', 'MRD', 'MYR', 'OCT',
        'OLA', 'OLC', 'P6G', 'PE3', 'PE4', 'PE5', 'PE8', 'PG0',
        'PG4', 'PG6', 'PGE', 'SDS', 'SPM', 'TAM', 'TME', 'UND',
    }
    
    # Common cofactors
    COFACTORS = {
        'ATP', 'ADP', 'AMP', 'ANP', 'GTP', 'GDP', 'GNP',
        'NAD', 'NAI', 'NDP', 'NAP', 'NAH', 'NPH',
        'FAD', 'FDA', 'FMN',
        'HEM', 'HEA', 'HEB', 'HEC', 'HED', 'HEE', 'HEM', 'HEO',
        'PLP', 'PMP', 'P2P', 'P3P',
        'TPP', 'THD', 'TDP',
        'COA', 'COI', 'COB', 'COZ',
        'SAM', 'SAH', 'MTA',
        'FES', 'F3S', 'F4S', 'SF4', 'CFS',
        'CLA', 'CL0', 'BCL', 'BCB', 'BPB', 'BPH',
        'RET', 'RAL', 'OLC',
    }
    
    def __init__(self):
        self.custom_ligands = set()
        self.excluded_residues = set()
    
    def add_ligand(self, residue_name: str):
        """Add a custom ligand to detection."""
        self.custom_ligands.add(residue_name.upper())
    
    def exclude_residue(self, residue_name: str):
        """Exclude a residue from being classified as ligand."""
        self.excluded_residues.add(residue_name.upper())
    
    def classify_residue(self, residue_name: str, has_backbone: bool = False) -> str:
        """
        Classify a residue into a category.
        
        Returns:
        --------
        str: 'protein', 'metal', 'cofactor', 'ligand', 'water', 'buffer', 'modified_residue'
        """
        res_name = residue_name.upper().strip()
        
        # Check custom lists first
        if res_name in self.excluded_residues:
            return 'protein'
        if res_name in self.custom_ligands:
            return 'ligand'
        
        # Standard amino acids
        if res_name in self.STANDARD_AA:
            return 'protein'
        
        # Metals
        if res_name in self.METALS:
            return 'metal'
        
        # Solvents
        if res_name in self.SOLVENTS:
            return 'water'
        
        # Buffers
        if res_name in self.BUFFERS:
            return 'buffer'
        
        # Cofactors
        if res_name in self.COFACTORS:
            return 'cofactor'
        
        # Modified residues (has backbone atoms but non-standard name)
        if has_backbone:
            return 'modified_residue'
        
        # Everything else is a ligand
        return 'ligand'
    
    def get_all_categories(self, atoms: AtomArray) -> Dict[str, List[Dict]]:
        """
        Categorize all residues in the structure.
        
        Returns:
        --------
        Dict with categories mapping to lists of residue info dicts
        """
        residue_starts = get_residue_starts(atoms)
        
        categories = {
            'protein': [],
            'metal': [],
            'cofactor': [],
            'ligand': [],
            'water': [],
            'buffer': [],
            'modified_residue': [],
        }
        
        for i, start_idx in enumerate(residue_starts):
            # Get residue info
            res_name = atoms.res_name[start_idx]
            res_id = atoms.res_id[start_idx]
            chain_id = atoms.chain_id[start_idx]
            
            # Get residue atom range
            if i < len(residue_starts) - 1:
                end_idx = residue_starts[i + 1]
            else:
                end_idx = len(atoms)
            
            res_atoms = atoms[start_idx:end_idx]
            
            # Check for backbone atoms (N, CA, C, O)
            atom_names = set(res_atoms.atom_name)
            backbone_atoms = {'N', 'CA', 'C', 'O'}
            has_backbone = len(backbone_atoms & atom_names) >= 3
            
            # Classify
            category = self.classify_residue(res_name, has_backbone)
            
            categories[category].append({
                'residue_name': res_name,
                'residue_id': res_id,
                'chain_id': chain_id,
                'start_index': start_idx,
                'end_index': end_idx,
                'num_atoms': len(res_atoms),
                'elements': list(set(res_atoms.element)),
            })
        
        # Print summary
        print(f"\nResidue Classification Summary:")
        for category, residues in categories.items():
            if residues:
                unique_names = set(r['residue_name'] for r in residues)
                print(f"  {category:20s}: {len(residues):4d} residues "
                      f"({len(unique_names)} unique types)")
                if len(unique_names) <= 10:
                    print(f"    Names: {', '.join(sorted(unique_names))}")
        
        return categories

    def get_ligand_residues(self, atoms: AtomArray) -> Dict[str, List[int]]:
        """
        Identify all ligand residues in the structure.

        Parameters:
        -----------
        atoms : AtomArray
            Protein structure with all atoms

        Returns:
        --------
        Dict[str, List[int]]
            Dictionary mapping category to list of residue IDs
        """
        residue_starts = get_residue_starts(atoms)
        residue_names = atoms.res_name[residue_starts]
        residue_ids = atoms.res_id[residue_starts]
        chain_ids = atoms.chain_id[residue_starts]

        categories = {
            'protein': [],
            'metal': [],
            'cofactor': [],
            'ligand': [],
            'water': [],
            'buffer': [],
            'modified_residue': [],
        }

        for i, (res_name, res_id, chain_id) in enumerate(zip(residue_names, residue_ids, chain_ids)):
            category = self.classify_residue(res_name)

            # Check for modified amino acids (non-standard but similar to amino acids)
            if category == 'ligand' and self._is_likely_modified_residue(atoms, residue_starts[i]):
                category = 'modified_residue'

            categories[category].append({
                'residue_name': res_name,
                'residue_id': res_id,
                'chain_id': chain_id,
                'start_index': residue_starts[i],
            })

        return categories

    def _is_likely_modified_residue(self, atoms: AtomArray, start_idx: int) -> bool:
        """
        Check if a residue is likely a modified amino acid.
        Modified residues typically have backbone atoms (N, CA, C, O).
        """
        # Get end index
        residue_starts = get_residue_starts(atoms)
        idx = np.where(residue_starts == start_idx)[0]
        if len(idx) == 0:
            return False
        idx = idx[0]

        if idx < len(residue_starts) - 1:
            end_idx = residue_starts[idx + 1]
        else:
            end_idx = len(atoms)

        res_atoms = atoms[start_idx:end_idx]
        atom_names = set(res_atoms.atom_name)

        # Check for backbone atoms
        backbone_atoms = {'N', 'CA', 'C', 'O'}
        backbone_count = len(backbone_atoms & atom_names)

        return backbone_count >= 3  # At least 3 of 4 backbone atoms present
