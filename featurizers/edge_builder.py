import numpy as np
from typing import List, Dict, Tuple, Union
from collections import defaultdict

from .definitions import BondType

from biotite.structure import AtomArray, AtomArrayStack, get_residue_starts


class ProteinLigandEdgeBuilder:
    """
    Complete edge builder for protein-ligand complexes.
    
    Builds all types of edges:
    - Intra-protein: covalent bonds, peptide bonds, disulfide bridges
    - Intra-ligand: covalent bonds within ligand molecules
    - Protein-ligand: covalent bonds, H-bonds, hydrophobic, ionic, pi-stacking
    - Metal coordination: protein-metal, ligand-metal, water-metal
    - Water-mediated: water bridges between protein and ligand
    """
    
    def __init__(self, 
                 covalent_cutoff_scale: float = 1.15,
                 hbond_distance_range: Tuple[float, float] = (2.5, 3.5),
                 hbond_angle_min: float = 120.0,
                 hydrophobic_distance_range: Tuple[float, float] = (3.0, 4.5),
                 ionic_distance_cutoff: float = 4.5,
                 pi_stack_distance_range: Tuple[float, float] = (3.5, 5.5),
                 halogen_distance_range: Tuple[float, float] = (2.5, 3.5),
                 metal_coordination_distances: Dict[str, Tuple[float, float]] = None,
                 include_water_bridges: bool = True,
                 water_bridge_distance: float = 3.2):
        
        self.covalent_cutoff_scale = covalent_cutoff_scale
        self.hbond_distance_range = hbond_distance_range
        self.hbond_angle_min = hbond_angle_min
        self.hydrophobic_distance_range = hydrophobic_distance_range
        self.ionic_distance_cutoff = ionic_distance_cutoff
        self.pi_stack_distance_range = pi_stack_distance_range
        self.halogen_distance_range = halogen_distance_range
        self.include_water_bridges = include_water_bridges
        self.water_bridge_distance = water_bridge_distance
        
        # Metal coordination distances (min, max) in Å
        self.metal_coordination_distances = metal_coordination_distances or {
            'MG': (1.8, 2.5), 'CA': (2.0, 2.8), 'ZN': (1.8, 2.5),
            'FE': (1.8, 2.6), 'MN': (1.8, 2.6), 'CU': (1.8, 2.6),
            'CO': (1.8, 2.5), 'NI': (1.8, 2.5), 'NA': (2.0, 3.0),
            'K': (2.5, 3.5), 'CD': (2.0, 2.8), 'HG': (2.0, 2.8),
        }
        
        # Covalent bond thresholds by element pair (in Å)
        self.covalent_thresholds = {
            ('C', 'C'): 1.65, ('C', 'N'): 1.55, ('C', 'O'): 1.55,
            ('C', 'S'): 1.90, ('C', 'P'): 1.95, ('C', 'F'): 1.45,
            ('C', 'Cl'): 1.85, ('C', 'Br'): 2.05, ('C', 'I'): 2.25,
            ('N', 'N'): 1.55, ('N', 'O'): 1.55, ('N', 'S'): 1.85,
            ('N', 'P'): 1.85, ('O', 'O'): 1.55, ('O', 'S'): 1.80,
            ('O', 'P'): 1.75, ('S', 'S'): 2.20, ('S', 'P'): 2.20,
            ('P', 'P'): 2.30, ('P', 'F'): 1.65, ('P', 'Cl'): 2.10,
            ('C', 'H'): 1.20, ('N', 'H'): 1.15, ('O', 'H'): 1.10,
            ('S', 'H'): 1.45, ('P', 'H'): 1.55,
        }
        
        # H-bond donor definitions
        self.hbond_donors = {
            'ARG': {'NE': 0.9, 'NH1': 1.0, 'NH2': 1.0},
            'LYS': {'NZ': 1.0},
            'HIS': {'ND1': 0.7, 'NE2': 0.7},
            'TRP': {'NE1': 0.8},
            'ASN': {'ND2': 0.8},
            'GLN': {'NE2': 0.8},
            'SER': {'OG': 0.6},
            'THR': {'OG1': 0.6},
            'TYR': {'OH': 0.8},
            'CYS': {'SG': 0.3},
            # Backbone NH
            '*': {'N': 0.5},
        }
        
        # H-bond acceptor definitions
        self.hbond_acceptors = {
            'ASP': {'OD1': 1.0, 'OD2': 1.0},
            'GLU': {'OE1': 1.0, 'OE2': 1.0},
            'ASN': {'OD1': 0.8},
            'GLN': {'OE1': 0.8},
            'HIS': {'ND1': 0.7, 'NE2': 0.7},
            'SER': {'OG': 0.6},
            'THR': {'OG1': 0.6},
            'TYR': {'OH': 0.6},
            'MET': {'SD': 0.3},
            'CYS': {'SG': 0.3},
            # Backbone C=O
            '*': {'O': 0.7},
        }
        
        # Standard amino acids
        self.standard_amino_acids = {
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
            'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
            'THR', 'TRP', 'TYR', 'VAL'
        }
        
        # Metal atoms
        self.metal_elements = {
            'MG', 'CA', 'ZN', 'FE', 'MN', 'CU', 'CO', 'NI', 'CD', 'HG',
            'NA', 'K', 'LI', 'RB', 'CS', 'BA', 'SR', 'PT', 'AU', 'AG',
            'MO', 'W', 'V', 'CR', 'AL',
        }
        
        # Coordination-capable atoms
        self.coordination_atoms = {
            'O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH',
            'N', 'ND1', 'NE2', 'NZ', 'NH1', 'NH2',
            'S', 'SD', 'SG',
        }
        
        # Hydrophobic residue classification
        self.hydrophobic_residues = {
            'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO', 'CYS'
        }
        
        # Aromatic residues
        self.aromatic_residues = {'PHE', 'TYR', 'TRP', 'HIS'}
        
        # Aromatic ring atoms
        self.aromatic_atoms = {
            'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'ND1', 'NE2', 'CE3', 'CZ2', 'CZ3', 'CH2'
        }
        
        # Charged residues
        self.positive_residues = {'LYS', 'ARG', 'HIS'}
        self.negative_residues = {'ASP', 'GLU'}
        
        # Halogen elements
        self.halogen_elements = {'F', 'Cl', 'Br', 'I'}
        
        # Halogen bond acceptors
        self.halogen_acceptors = {'O', 'N', 'S'}
    
    def build_complete_edge_graph(self, atoms: Union[AtomArray, AtomArrayStack]) -> Dict[str, List[Dict]]:
        """
        Build complete edge graph for protein-ligand complex.
        
        Parameters:
        -----------
        atoms : AtomArray or AtomArrayStack
            Complete structure with protein, ligands, metals, water
            
        Returns:
        --------
        Dict[str, List[Dict]]
            Categorized edges:
            - covalent: Intra-residue covalent bonds
            - peptide: Protein backbone peptide bonds
            - disulfide: Cysteine disulfide bridges
            - aromatic: Aromatic ring bonds
            - intra_ligand: Bonds within ligand molecules
            - ligand_covalent: Covalent bonds between protein and ligand
            - metal_coordination: Metal coordination bonds
            - hydrogen_bond: All hydrogen bonds
            - ionic: Ionic interactions
            - pi_stacking: pi-pi stacking interactions
            - hydrophobic: Hydrophobic contacts
            - halogen_bond: Halogen bonds
            - van_der_waals: Van der Waals contacts
        """
        # Handle AtomArrayStack by taking first model
        if isinstance(atoms, AtomArrayStack):
            if len(atoms) > 1:
                print(f"AtomArrayStack with {len(atoms)} models, using model 0")
            atoms = atoms[0]
        
        n_atoms = len(atoms)
        
        # Initialize edge categories
        edges = {
            'covalent': [],
            'peptide': [],
            'disulfide': [],
            'aromatic': [],
            'intra_ligand': [],
            'ligand_covalent': [],
            'metal_coordination': [],
            'hydrogen_bond': [],
            'ionic': [],
            'pi_stacking': [],
            'hydrophobic': [],
            'halogen_bond': [],
            'van_der_waals': [],
        }
        
        # Get residue boundaries
        residue_starts = get_residue_starts(atoms)
        
        # Classify residues
        residue_info = self._classify_residues(atoms, residue_starts)
        
        # Build a mapping from global atom index to residue info
        atom_to_residue = self._build_atom_to_residue_map(atoms, residue_starts, residue_info)
        
        # 1. Build intra-residue covalent bonds
        self._build_intra_residue_bonds(atoms, residue_starts, residue_info, edges)
        
        # 2. Build peptide bonds between consecutive protein residues
        self._build_peptide_bonds(atoms, residue_starts, residue_info, edges)
        
        # 3. Build disulfide bonds
        self._build_disulfide_bonds(atoms, residue_starts, residue_info, edges)
        
        # 4. Build intra-ligand bonds
        self._build_intra_ligand_bonds(atoms, residue_starts, residue_info, edges)
        
        # 5. Detect protein-ligand covalent bonds
        self._build_protein_ligand_covalent_bonds(atoms, residue_starts, residue_info, edges)
        
        # 6. Build metal coordination
        self._build_metal_coordination(atoms, residue_starts, residue_info, edges)
        
        # 7. Build non-covalent interactions
        self._build_non_covalent_interactions(atoms, residue_starts, residue_info, edges)
        
        return edges
    
    def _classify_residues(self, atoms: AtomArray, residue_starts: np.ndarray) -> Dict[str, List[Dict]]:
        """Classify all residues into categories."""
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
            # Get end index
            if i < len(residue_starts) - 1:
                end_idx = residue_starts[i + 1]
            else:
                end_idx = len(atoms)
            
            res_name = atoms.res_name[start_idx]
            res_id = atoms.res_id[start_idx]
            chain_id = atoms.chain_id[start_idx]
            res_atoms = atoms[start_idx:end_idx]
            
            # Check for backbone atoms
            atom_names = set(res_atoms.atom_name)
            backbone_atoms = {'N', 'CA', 'C', 'O'}
            has_backbone = len(backbone_atoms & atom_names) >= 3
            
            # Classify
            if res_name in self.standard_amino_acids:
                category = 'protein'
            elif res_name in self.metal_elements:
                category = 'metal'
            elif res_name in {'HOH', 'WAT', 'H2O', 'DOD'}:
                category = 'water'
            elif has_backbone:
                category = 'modified_residue'
            else:
                # Check if single atom (likely ion/metal)
                if len(res_atoms) == 1:
                    elem = res_atoms.element[0]
                    if elem in self.metal_elements:
                        category = 'metal'
                    else:
                        category = 'ligand'
                else:
                    category = 'ligand'
            
            categories[category].append({
                'residue_name': res_name,
                'residue_id': res_id,
                'chain_id': chain_id,
                'start_index': start_idx,
                'end_index': end_idx,
                'num_atoms': len(res_atoms),
            })
        
        return categories
    
    def _build_atom_to_residue_map(self, atoms: AtomArray, residue_starts: np.ndarray,
                                   residue_info: Dict[str, List[Dict]]) -> List[Dict]:
        """Build mapping from global atom index to residue info."""
        atom_map = [None] * len(atoms)
        
        for category, residues in residue_info.items():
            for res_info in residues:
                for idx in range(res_info['start_index'], res_info['end_index']):
                    atom_map[idx] = {
                        'category': category,
                        'residue_name': res_info['residue_name'],
                        'residue_id': res_info['residue_id'],
                        'chain_id': res_info['chain_id'],
                        'start_index': res_info['start_index'],
                    }
        
        return atom_map
    
    def _build_intra_residue_bonds(self, atoms: AtomArray, residue_starts: np.ndarray,
                                   residue_info: Dict[str, List[Dict]],
                                   edges: Dict[str, List[Dict]]):
        """Build covalent bonds within residues."""
        for category in ['protein', 'modified_residue']:
            for res_info in residue_info.get(category, []):
                start_idx = res_info['start_index']
                end_idx = res_info['end_index']
                res_atoms = atoms[start_idx:end_idx]
                
                self._detect_covalent_bonds_in_group(
                    res_atoms, start_idx, res_info, 'covalent', edges, 
                    is_aromatic=(res_info['residue_name'] in self.aromatic_residues)
                )
    
    def _build_intra_ligand_bonds(self, atoms: AtomArray, residue_starts: np.ndarray,
                                  residue_info: Dict[str, List[Dict]],
                                  edges: Dict[str, List[Dict]]):
        """Build covalent bonds within ligand molecules."""
        for category in ['ligand', 'cofactor']:
            for res_info in residue_info.get(category, []):
                start_idx = res_info['start_index']
                end_idx = res_info['end_index']
                res_atoms = atoms[start_idx:end_idx]
                
                self._detect_covalent_bonds_in_group(
                    res_atoms, start_idx, res_info, 'intra_ligand', edges,
                    is_aromatic=False
                )
    
    def _detect_covalent_bonds_in_group(self, group_atoms: AtomArray, start_idx: int,
                                        res_info: Dict, category: str,
                                        edges: Dict[str, List[Dict]],
                                        is_aromatic: bool = False):
        """Detect covalent bonds within a group of atoms."""
        atom_names = group_atoms.atom_name
        elements = group_atoms.element
        coords = group_atoms.coord
        n_atoms = len(group_atoms)
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                elem1, elem2 = elements[i], elements[j]
                dist = np.linalg.norm(coords[i] - coords[j])
                
                # Get covalent threshold
                max_dist = 0.0
                for (e1, e2), threshold in self.covalent_thresholds.items():
                    if (elem1 == e1 and elem2 == e2) or (elem1 == e2 and elem2 == e1):
                        max_dist = threshold * self.covalent_cutoff_scale
                        break
                
                if max_dist == 0.0:
                    # Default: sum of covalent radii approximation
                    max_dist = 2.0
                
                if dist <= max_dist:
                    # Determine bond type
                    if is_aromatic and self._is_aromatic_bond(atom_names[i], atom_names[j]):
                        bond_type = BondType.AROMATIC_DOUBLE if np.random.random() > 0.5 else BondType.AROMATIC_SINGLE
                    elif dist < 1.30:
                        bond_type = BondType.DOUBLE
                    elif dist < 1.20:
                        bond_type = BondType.TRIPLE
                    else:
                        bond_type = BondType.SINGLE
                    
                    edge = {
                        'atom1': f"{res_info['residue_name']}{res_info['residue_id']}_{atom_names[i]}",
                        'atom2': f"{res_info['residue_name']}{res_info['residue_id']}_{atom_names[j]}",
                        'atom1_name': atom_names[i],
                        'atom2_name': atom_names[j],
                        'atom1_element': elements[i],
                        'atom2_element': elements[j],
                        'atom1_index': int(start_idx + i),
                        'atom2_index': int(start_idx + j),
                        'atom1_local_index': int(i),
                        'atom2_local_index': int(j),
                        'chain': res_info['chain_id'],
                        'res_id': res_info['residue_id'],
                        'res_name': res_info['residue_name'],
                        'bond_type': bond_type,
                        'bond_type_value': int(bond_type),
                        'bond_type_name': bond_type.name,
                        'distance': float(dist),
                        'bond_order': bond_type.bond_order,
                        'is_covalent': True,
                        'edge_category': category,
                    }
                    
                    # Categorize
                    if bond_type.is_aromatic:
                        edges['aromatic'].append(edge)
                    else:
                        edges[category].append(edge)
    
    def _is_aromatic_bond(self, atom_name1: str, atom_name2: str) -> bool:
        """Check if two atom names form an aromatic bond."""
        aromatic_pairs = {
            ('CG', 'CD1'), ('CG', 'CD2'), ('CD1', 'CE1'), ('CD2', 'CE2'),
            ('CE1', 'CZ'), ('CE2', 'CZ'), ('CG', 'ND1'), ('ND1', 'CE1'),
            ('CE1', 'NE2'), ('CD2', 'NE2'), ('CD2', 'CE2'), ('CD2', 'CE3'),
            ('CE2', 'CZ2'), ('CE3', 'CZ3'), ('CZ2', 'CH2'), ('CZ3', 'CH2'),
            ('NE1', 'CE2'), ('CG', 'CD2'),
        }
        return (atom_name1, atom_name2) in aromatic_pairs or (atom_name2, atom_name1) in aromatic_pairs
    
    def _build_peptide_bonds(self, atoms: AtomArray, residue_starts: np.ndarray,
                             residue_info: Dict[str, List[Dict]],
                             edges: Dict[str, List[Dict]]):
        """Build peptide bonds between consecutive protein residues."""
        protein_residues = residue_info.get('protein', [])
        
        # Sort by start index
        protein_residues = sorted(protein_residues, key=lambda x: x['start_index'])
        
        for i in range(len(protein_residues) - 1):
            res1 = protein_residues[i]
            res2 = protein_residues[i + 1]
            
            # Check same chain and consecutive residue IDs
            if res1['chain_id'] != res2['chain_id']:
                continue
            
            # Get C from res1 and N from res2
            atoms1 = atoms[res1['start_index']:res1['end_index']]
            atoms2 = atoms[res2['start_index']:res2['end_index']]
            
            c_mask = (atoms1.atom_name == 'C')
            n_mask = (atoms2.atom_name == 'N')
            
            if np.any(c_mask) and np.any(n_mask):
                c_idx = np.where(c_mask)[0][0]
                n_idx = np.where(n_mask)[0][0]
                
                dist = np.linalg.norm(atoms1.coord[c_idx] - atoms2.coord[n_idx])
                
                # Peptide bond length ~1.33 Å
                if dist < 1.6:
                    edge = {
                        'atom1': f"{res1['residue_name']}{res1['residue_id']}_C",
                        'atom2': f"{res2['residue_name']}{res2['residue_id']}_N",
                        'atom1_name': 'C',
                        'atom2_name': 'N',
                        'atom1_element': 'C',
                        'atom2_element': 'N',
                        'atom1_index': int(res1['start_index'] + c_idx),
                        'atom2_index': int(res2['start_index'] + n_idx),
                        'prev_res_name': res1['residue_name'],
                        'curr_res_name': res2['residue_name'],
                        'prev_res_id': res1['residue_id'],
                        'curr_res_id': res2['residue_id'],
                        'bond_type': BondType.PEPTIDE,
                        'bond_type_value': int(BondType.PEPTIDE),
                        'bond_type_name': BondType.PEPTIDE.name,
                        'distance': float(dist),
                        'bond_order': BondType.PEPTIDE.bond_order,
                        'is_covalent': True,
                        'edge_category': 'peptide',
                        'description': f"Peptide bond: {res1['residue_name']}{res1['residue_id']}-{res2['residue_name']}{res2['residue_id']}",
                    }
                    edges['peptide'].append(edge)
    
    def _build_disulfide_bonds(self, atoms: AtomArray, residue_starts: np.ndarray,
                               residue_info: Dict[str, List[Dict]],
                               edges: Dict[str, List[Dict]]):
        """Build disulfide bonds between cysteine residues."""
        protein_residues = residue_info.get('protein', [])
        cys_residues = [r for r in protein_residues if r['residue_name'] == 'CYS']
        
        for i in range(len(cys_residues)):
            for j in range(i + 1, len(cys_residues)):
                res1 = cys_residues[i]
                res2 = cys_residues[j]
                
                atoms1 = atoms[res1['start_index']:res1['end_index']]
                atoms2 = atoms[res2['start_index']:res2['end_index']]
                
                sg1_mask = (atoms1.atom_name == 'SG')
                sg2_mask = (atoms2.atom_name == 'SG')
                
                if np.any(sg1_mask) and np.any(sg2_mask):
                    sg1_idx = np.where(sg1_mask)[0][0]
                    sg2_idx = np.where(sg2_mask)[0][0]
                    
                    dist = np.linalg.norm(atoms1.coord[sg1_idx] - atoms2.coord[sg2_idx])
                    
                    if 1.8 <= dist <= 2.2:
                        edge = {
                            'atom1': f"CYS{res1['residue_id']}_SG",
                            'atom2': f"CYS{res2['residue_id']}_SG",
                            'atom1_name': 'SG',
                            'atom2_name': 'SG',
                            'atom1_element': 'S',
                            'atom2_element': 'S',
                            'atom1_index': int(res1['start_index'] + sg1_idx),
                            'atom2_index': int(res2['start_index'] + sg2_idx),
                            'bond_type': BondType.DISULFIDE,
                            'bond_type_value': int(BondType.DISULFIDE),
                            'bond_type_name': BondType.DISULFIDE.name,
                            'distance': float(dist),
                            'is_covalent': True,
                            'edge_category': 'disulfide',
                            'description': f"Disulfide: CYS{res1['residue_id']}-CYS{res2['residue_id']}",
                        }
                        edges['disulfide'].append(edge)
    
    def _build_protein_ligand_covalent_bonds(self, atoms: AtomArray, residue_starts: np.ndarray,
                                             residue_info: Dict[str, List[Dict]],
                                             edges: Dict[str, List[Dict]]):
        """Detect covalent bonds between protein and ligand."""
        protein_residues = residue_info.get('protein', []) + residue_info.get('modified_residue', [])
        ligand_residues = residue_info.get('ligand', []) + residue_info.get('cofactor', [])
        
        for prot_res in protein_residues:
            prot_atoms = atoms[prot_res['start_index']:prot_res['end_index']]
            
            for lig_res in ligand_residues:
                lig_atoms = atoms[lig_res['start_index']:lig_res['end_index']]
                
                for i in range(len(prot_atoms)):
                    for j in range(len(lig_atoms)):
                        elem1 = prot_atoms.element[i]
                        elem2 = lig_atoms.element[j]
                        dist = np.linalg.norm(prot_atoms.coord[i] - lig_atoms.coord[j])
                        
                        # Check covalent threshold
                        max_dist = 0.0
                        for (e1, e2), threshold in self.covalent_thresholds.items():
                            if (elem1 == e1 and elem2 == e2) or (elem1 == e2 and elem2 == e1):
                                max_dist = threshold * self.covalent_cutoff_scale
                                break
                        
                        if max_dist > 0 and dist <= max_dist:
                            edge = {
                                'atom1': f"{prot_res['residue_name']}{prot_res['residue_id']}_{prot_atoms.atom_name[i]}",
                                'atom2': f"{lig_res['residue_name']}_{lig_atoms.atom_name[j]}",
                                'atom1_name': prot_atoms.atom_name[i],
                                'atom2_name': lig_atoms.atom_name[j],
                                'atom1_element': elem1,
                                'atom2_element': elem2,
                                'atom1_index': int(prot_res['start_index'] + i),
                                'atom2_index': int(lig_res['start_index'] + j),
                                'res1_name': prot_res['residue_name'],
                                'res2_name': lig_res['residue_name'],
                                'res1_id': prot_res['residue_id'],
                                'res2_id': lig_res['residue_id'],
                                'bond_type': BondType.LIGAND_COVALENT,
                                'bond_type_value': int(BondType.LIGAND_COVALENT),
                                'bond_type_name': BondType.LIGAND_COVALENT.name,
                                'distance': float(dist),
                                'is_covalent': True,
                                'edge_category': 'ligand_covalent',
                                'description': f"Covalent: {prot_res['residue_name']}-{lig_res['residue_name']}",
                            }
                            edges['ligand_covalent'].append(edge)
    
    def _build_metal_coordination(self, atoms: AtomArray, residue_starts: np.ndarray,
                                  residue_info: Dict[str, List[Dict]],
                                  edges: Dict[str, List[Dict]]):
        """Build metal coordination bonds."""
        metal_residues = residue_info.get('metal', [])
        
        # All non-metal residues
        other_residues = []
        for cat in ['protein', 'modified_residue', 'ligand', 'cofactor', 'water']:
            other_residues.extend(residue_info.get(cat, []))
        
        for metal_res in metal_residues:
            metal_atoms = atoms[metal_res['start_index']:metal_res['end_index']]
            metal_name = metal_res['residue_name'].upper()
            
            # Get distance range for this metal
            min_dist, max_dist = self.metal_coordination_distances.get(
                metal_name, (1.8, 2.8)
            )
            
            for other_res in other_residues:
                if other_res['start_index'] == metal_res['start_index']:
                    continue  # Same residue
                
                other_atoms = atoms[other_res['start_index']:other_res['end_index']]
                
                for mi in range(len(metal_atoms)):
                    metal_coord = metal_atoms.coord[mi]
                    metal_elem = metal_atoms.element[mi]
                    metal_atom_name = metal_atoms.atom_name[mi]
                    
                    for oi in range(len(other_atoms)):
                        other_coord = other_atoms.coord[oi]
                        other_elem = other_atoms.element[oi]
                        other_name = other_atoms.atom_name[oi]
                        
                        dist = np.linalg.norm(metal_coord - other_coord)
                        
                        if min_dist <= dist <= max_dist:
                            # Check if atom can coordinate
                            can_coordinate = (
                                other_name in self.coordination_atoms or
                                other_elem in {'O', 'N', 'S'}
                            )
                            
                            if can_coordinate:
                                edge = {
                                    'atom1': f"{metal_name}_{metal_atom_name}",
                                    'atom2': f"{other_res['residue_name']}_{other_name}",
                                    'atom1_name': metal_atom_name,
                                    'atom2_name': other_name,
                                    'atom1_element': metal_elem,
                                    'atom2_element': other_elem,
                                    'atom1_index': int(metal_res['start_index'] + mi),
                                    'atom2_index': int(other_res['start_index'] + oi),
                                    'res1_name': metal_name,
                                    'res2_name': other_res['residue_name'],
                                    'res1_category': 'metal',
                                    'res2_category': other_res.get('category', 'unknown'),
                                    'bond_type': BondType.METAL_COORDINATION,
                                    'bond_type_value': int(BondType.METAL_COORDINATION),
                                    'bond_type_name': BondType.METAL_COORDINATION.name,
                                    'distance': float(dist),
                                    'is_covalent': False,
                                    'edge_category': 'metal_coordination',
                                    'description': f"Metal coord: {metal_name}-{other_res['residue_name']}({other_name})",
                                }
                                edges['metal_coordination'].append(edge)
    
    def _build_non_covalent_interactions(self, atoms: AtomArray, residue_starts: np.ndarray,
                                         residue_info: Dict[str, List[Dict]],
                                         edges: Dict[str, List[Dict]]):
        """Build all non-covalent interactions."""
        
        # Collect all non-water residues
        all_residues = []
        for cat in ['protein', 'modified_residue', 'ligand', 'cofactor', 'metal']:
            for res in residue_info.get(cat, []):
                res['category'] = cat
                all_residues.append(res)
        
        # Check all residue pairs
        for i in range(len(all_residues)):
            for j in range(i + 1, len(all_residues)):
                res1 = all_residues[i]
                res2 = all_residues[j]
                
                # Skip if adjacent protein residues (to avoid backbone interactions)
                if (res1['category'] == 'protein' and res2['category'] == 'protein' and
                    res1['chain_id'] == res2['chain_id'] and
                    abs(res1['residue_id'] - res2['residue_id']) <= 2):
                    continue
                
                atoms1 = atoms[res1['start_index']:res1['end_index']]
                atoms2 = atoms[res2['start_index']:res2['end_index']]
                
                # Detect H-bonds
                self._detect_hydrogen_bonds(atoms1, atoms2, res1, res2, edges)
                
                # Detect ionic interactions
                self._detect_ionic_interactions(atoms1, atoms2, res1, res2, edges)
                
                # Detect hydrophobic contacts
                self._detect_hydrophobic_contacts(atoms1, atoms2, res1, res2, edges)
                
                # Detect pi-stacking
                self._detect_pi_stacking(atoms1, atoms2, res1, res2, edges)
                
                # Detect halogen bonds
                self._detect_halogen_bonds(atoms1, atoms2, res1, res2, edges)
    
    def _detect_hydrogen_bonds(self, atoms1: AtomArray, atoms2: AtomArray,
                               res1: Dict, res2: Dict, edges: Dict[str, List[Dict]]):
        """Detect hydrogen bonds between two residues."""
        min_dist, max_dist = self.hbond_distance_range
        
        # Get donors and acceptors
        donors1 = self._get_donors(atoms1, res1)
        acceptors1 = self._get_acceptors(atoms1, res1)
        donors2 = self._get_donors(atoms2, res2)
        acceptors2 = self._get_acceptors(atoms2, res2)
        
        # Check donor1-acceptor2
        for d_idx, d_name, d_elem in donors1:
            d_coord = atoms1.coord[d_idx]
            for a_idx, a_name, a_elem in acceptors2:
                a_coord = atoms2.coord[a_idx]
                dist = np.linalg.norm(d_coord - a_coord)
                
                if min_dist <= dist <= max_dist:
                    is_ligand_involved = (
                        res1.get('category') in ['ligand', 'cofactor'] or
                        res2.get('category') in ['ligand', 'cofactor']
                    )
                    
                    edge = self._make_non_covalent_edge(
                        atoms1, atoms2, d_idx, a_idx, res1, res2, dist,
                        BondType.LIGAND_HYDROGEN if is_ligand_involved else BondType.HYDROGEN,
                        'hydrogen_bond',
                        f"H-bond: {res1['residue_name']}({d_name})···{res2['residue_name']}({a_name})"
                    )
                    edges['hydrogen_bond'].append(edge)
        
        # Check donor2-acceptor1
        for d_idx, d_name, d_elem in donors2:
            d_coord = atoms2.coord[d_idx]
            for a_idx, a_name, a_elem in acceptors1:
                a_coord = atoms1.coord[a_idx]
                dist = np.linalg.norm(d_coord - a_coord)
                
                if min_dist <= dist <= max_dist:
                    is_ligand_involved = (
                        res1.get('category') in ['ligand', 'cofactor'] or
                        res2.get('category') in ['ligand', 'cofactor']
                    )
                    
                    edge = self._make_non_covalent_edge(
                        atoms2, atoms1, d_idx, a_idx, res2, res1, dist,
                        BondType.LIGAND_HYDROGEN if is_ligand_involved else BondType.HYDROGEN,
                        'hydrogen_bond',
                        f"H-bond: {res2['residue_name']}({d_name})···{res1['residue_name']}({a_name})"
                    )
                    edges['hydrogen_bond'].append(edge)
    
    def _get_donors(self, atoms: AtomArray, res_info: Dict) -> List[Tuple[int, str, str]]:
        """Get H-bond donor atoms from residue."""
        donors = []
        res_name = res_info['residue_name']
        
        # Check pre-defined donors
        donor_dict = self.hbond_donors.get(res_name, {})
        universal_donors = self.hbond_donors.get('*', {})
        all_donors = {**donor_dict, **universal_donors}
        
        for idx, atom_name in enumerate(atoms.atom_name):
            if atom_name in all_donors:
                donors.append((idx, atom_name, atoms.element[idx]))
            # Also include N with hydrogen (backbone NH)
            elif atom_name == 'N' and res_info.get('category') == 'protein':
                donors.append((idx, atom_name, atoms.element[idx]))
        
        return donors
    
    def _get_acceptors(self, atoms: AtomArray, res_info: Dict) -> List[Tuple[int, str, str]]:
        """Get H-bond acceptor atoms from residue."""
        acceptors = []
        res_name = res_info['residue_name']
        
        # Check pre-defined acceptors
        acceptor_dict = self.hbond_acceptors.get(res_name, {})
        universal_acceptors = self.hbond_acceptors.get('*', {})
        all_acceptors = {**acceptor_dict, **universal_acceptors}
        
        for idx, atom_name in enumerate(atoms.atom_name):
            if atom_name in all_acceptors:
                acceptors.append((idx, atom_name, atoms.element[idx]))
            # Ligand heteroatoms
            elif res_info.get('category') in ['ligand', 'cofactor'] and atoms.element[idx] in {'O', 'N', 'S'}:
                acceptors.append((idx, atom_name, atoms.element[idx]))
        
        return acceptors
    
    def _detect_ionic_interactions(self, atoms1: AtomArray, atoms2: AtomArray,
                                   res1: Dict, res2: Dict, edges: Dict[str, List[Dict]]):
        """Detect ionic interactions between charged residues."""
        res1_name = res1['residue_name']
        res2_name = res2['residue_name']
        
        # Check if opposite charges
        is_ionic = (
            (res1_name in self.positive_residues and res2_name in self.negative_residues) or
            (res1_name in self.negative_residues and res2_name in self.positive_residues)
        )
        
        if not is_ionic:
            return
        
        # Get charged atoms
        positive_atoms = {'NZ', 'NH1', 'NH2', 'NE', 'ND1', 'NE2'}
        negative_atoms = {'OD1', 'OD2', 'OE1', 'OE2'}
        
        if res1_name in self.positive_residues:
            pos_atoms1 = [(i, n, atoms1.element[i]) for i, n in enumerate(atoms1.atom_name) if n in positive_atoms]
            neg_atoms2 = [(i, n, atoms2.element[i]) for i, n in enumerate(atoms2.atom_name) if n in negative_atoms]
        else:
            neg_atoms1 = [(i, n, atoms1.element[i]) for i, n in enumerate(atoms1.atom_name) if n in negative_atoms]
            pos_atoms2 = [(i, n, atoms2.element[i]) for i, n in enumerate(atoms2.atom_name) if n in positive_atoms]
        
        # Check distances
        if res1_name in self.positive_residues:
            for pi, pn, pe in pos_atoms1:
                for ni, nn, ne in neg_atoms2:
                    dist = np.linalg.norm(atoms1.coord[pi] - atoms2.coord[ni])
                    if dist <= self.ionic_distance_cutoff:
                        edge = self._make_non_covalent_edge(
                            atoms1, atoms2, pi, ni, res1, res2, dist,
                            BondType.IONIC, 'ionic',
                            f"Ionic: {res1_name}({pn})-{res2_name}({nn})"
                        )
                        edges['ionic'].append(edge)
        else:
            for ni, nn, ne in neg_atoms1:
                for pi, pn, pe in pos_atoms2:
                    dist = np.linalg.norm(atoms1.coord[ni] - atoms2.coord[pi])
                    if dist <= self.ionic_distance_cutoff:
                        edge = self._make_non_covalent_edge(
                            atoms1, atoms2, ni, pi, res1, res2, dist,
                            BondType.IONIC, 'ionic',
                            f"Ionic: {res1_name}({nn})-{res2_name}({pn})"
                        )
                        edges['ionic'].append(edge)
    
    def _detect_hydrophobic_contacts(self, atoms1: AtomArray, atoms2: AtomArray,
                                     res1: Dict, res2: Dict, edges: Dict[str, List[Dict]]):
        """Detect hydrophobic contacts between non-polar atoms."""
        res1_name = res1['residue_name']
        res2_name = res2['residue_name']
        
        # Check if either residue is hydrophobic
        is_hydrophobic = (
            res1_name in self.hydrophobic_residues or 
            res2_name in self.hydrophobic_residues or
            res1.get('category') in ['ligand', 'cofactor'] or
            res2.get('category') in ['ligand', 'cofactor']
        )
        
        if not is_hydrophobic:
            return
        
        # Get carbon atoms (hydrophobic)
        carbon1 = [(i, atoms1.element[i]) for i, e in enumerate(atoms1.element) if e == 'C']
        carbon2 = [(i, atoms2.element[i]) for i, e in enumerate(atoms2.element) if e == 'C']
        
        min_dist, max_dist = self.hydrophobic_distance_range
        
        # Count contacts
        contacts = 0
        contact_pairs = []
        
        for c1_idx, _ in carbon1:
            for c2_idx, _ in carbon2:
                dist = np.linalg.norm(atoms1.coord[c1_idx] - atoms2.coord[c2_idx])
                if min_dist <= dist <= max_dist:
                    contacts += 1
                    contact_pairs.append((c1_idx, c2_idx, dist))
        
        # Only create edge if significant contacts (>= 3)
        if contacts >= 3:
            # Average distance
            avg_dist = np.mean([d for _, _, d in contact_pairs])
            
            is_ligand = (
                res1.get('category') in ['ligand', 'cofactor'] or
                res2.get('category') in ['ligand', 'cofactor']
            )
            
            edge = {
                'atom1': f"{res1['residue_name']}{res1.get('residue_id', '')}",
                'atom2': f"{res2['residue_name']}{res2.get('residue_id', '')}",
                'res1_name': res1['residue_name'],
                'res2_name': res2['residue_name'],
                'res1_category': res1.get('category', 'unknown'),
                'res2_category': res2.get('category', 'unknown'),
                'bond_type': BondType.LIGAND_HYDROPHOBIC if is_ligand else BondType.HYDROPHOBIC,
                'bond_type_value': int(BondType.LIGAND_HYDROPHOBIC if is_ligand else BondType.HYDROPHOBIC),
                'bond_type_name': (BondType.LIGAND_HYDROPHOBIC if is_ligand else BondType.HYDROPHOBIC).name,
                'distance': float(avg_dist),
                'contact_count': contacts,
                'is_covalent': False,
                'edge_category': 'hydrophobic',
                'description': f"Hydrophobic: {res1['residue_name']}-{res2['residue_name']} ({contacts} contacts)",
            }
            edges['hydrophobic'].append(edge)
    
    def _detect_pi_stacking(self, atoms1: AtomArray, atoms2: AtomArray,
                            res1: Dict, res2: Dict, edges: Dict[str, List[Dict]]):
        """Detect pi-pi stacking between aromatic residues."""
        res1_name = res1['residue_name']
        res2_name = res2['residue_name']
        
        # Both must be aromatic
        if not (res1_name in self.aromatic_residues and res2_name in self.aromatic_residues):
            return
        
        # Get aromatic ring atoms
        arom1 = [(i, atoms1.coord[i]) for i, n in enumerate(atoms1.atom_name) if n in self.aromatic_atoms]
        arom2 = [(i, atoms2.coord[i]) for i, n in enumerate(atoms2.atom_name) if n in self.aromatic_atoms]
        
        if len(arom1) < 3 or len(arom2) < 3:
            return
        
        # Compute ring centroids
        centroid1 = np.mean([c for _, c in arom1], axis=0)
        centroid2 = np.mean([c for _, c in arom2], axis=0)
        
        dist = np.linalg.norm(centroid1 - centroid2)
        min_dist, max_dist = self.pi_stack_distance_range
        
        if min_dist <= dist <= max_dist:
            is_ligand = (
                res1.get('category') in ['ligand', 'cofactor'] or
                res2.get('category') in ['ligand', 'cofactor']
            )
            
            edge = {
                'atom1': f"{res1['residue_name']}{res1.get('residue_id', '')}_center",
                'atom2': f"{res2['residue_name']}{res2.get('residue_id', '')}_center",
                'res1_name': res1['residue_name'],
                'res2_name': res2['residue_name'],
                'res1_category': res1.get('category', 'unknown'),
                'res2_category': res2.get('category', 'unknown'),
                'bond_type': BondType.LIGAND_PI_STACKING if is_ligand else BondType.PI_STACKING,
                'bond_type_value': int(BondType.LIGAND_PI_STACKING if is_ligand else BondType.PI_STACKING),
                'bond_type_name': (BondType.LIGAND_PI_STACKING if is_ligand else BondType.PI_STACKING).name,
                'distance': float(dist),
                'is_covalent': False,
                'edge_category': 'pi_stacking',
                'description': f"Pi-stack: {res1['residue_name']}-{res2['residue_name']} ({dist:.2f} Å)",
            }
            edges['pi_stacking'].append(edge)
    
    def _detect_halogen_bonds(self, atoms1: AtomArray, atoms2: AtomArray,
                              res1: Dict, res2: Dict, edges: Dict[str, List[Dict]]):
        """Detect halogen bonds (C-X···O/N)."""
        min_dist, max_dist = self.halogen_distance_range
        
        # Find halogen atoms
        halogen1 = [(i, atoms1.element[i], atoms1.atom_name[i]) 
                   for i, e in enumerate(atoms1.element) if e in self.halogen_elements]
        halogen2 = [(i, atoms2.element[i], atoms2.atom_name[i]) 
                   for i, e in enumerate(atoms2.element) if e in self.halogen_elements]
        
        # Find acceptor atoms
        acceptors1 = [(i, atoms1.element[i]) for i, e in enumerate(atoms1.element) if e in self.halogen_acceptors]
        acceptors2 = [(i, atoms2.element[i]) for i, e in enumerate(atoms2.element) if e in self.halogen_acceptors]
        
        # Check halogen1-acceptor2
        for hi, he, hn in halogen1:
            for ai, ae in acceptors2:
                dist = np.linalg.norm(atoms1.coord[hi] - atoms2.coord[ai])
                if min_dist <= dist <= max_dist:
                    edge = self._make_non_covalent_edge(
                        atoms1, atoms2, hi, ai, res1, res2, dist,
                        BondType.HALOGEN, 'halogen_bond',
                        f"Halogen: {res1['residue_name']}({hn})-{res2['residue_name']}"
                    )
                    edges['halogen_bond'].append(edge)
        
        # Check halogen2-acceptor1
        for hi, he, hn in halogen2:
            for ai, ae in acceptors1:
                dist = np.linalg.norm(atoms2.coord[hi] - atoms1.coord[ai])
                if min_dist <= dist <= max_dist:
                    edge = self._make_non_covalent_edge(
                        atoms2, atoms1, hi, ai, res2, res1, dist,
                        BondType.HALOGEN, 'halogen_bond',
                        f"Halogen: {res2['residue_name']}({hn})-{res1['residue_name']}"
                    )
                    edges['halogen_bond'].append(edge)
    
    def _make_non_covalent_edge(self, atoms1: AtomArray, atoms2: AtomArray,
                                idx1: int, idx2: int,
                                res1: Dict, res2: Dict,
                                distance: float, bond_type: BondType,
                                category: str, description: str) -> Dict:
        """Create a non-covalent edge dictionary."""
        return {
            'atom1': f"{res1['residue_name']}_{atoms1.atom_name[idx1]}",
            'atom2': f"{res2['residue_name']}_{atoms2.atom_name[idx2]}",
            'atom1_name': atoms1.atom_name[idx1],
            'atom2_name': atoms2.atom_name[idx2],
            'atom1_element': atoms1.element[idx1],
            'atom2_element': atoms2.element[idx2],
            'atom1_index': int(res1['start_index'] + idx1),
            'atom2_index': int(res2['start_index'] + idx2),
            'res1_name': res1['residue_name'],
            'res2_name': res2['residue_name'],
            'res1_id': res1.get('residue_id', ''),
            'res2_id': res2.get('residue_id', ''),
            'res1_category': res1.get('category', 'unknown'),
            'res2_category': res2.get('category', 'unknown'),
            'bond_type': bond_type,
            'bond_type_value': int(bond_type),
            'bond_type_name': bond_type.name,
            'distance': float(distance),
            'is_covalent': False,
            'edge_category': category,
            'description': description,
        }
    
    def get_edge_statistics(self, edges: Dict[str, List[Dict]]) -> Dict:
        """Get statistics about the edge graph."""
        stats = {
            'total_edges': sum(len(v) for v in edges.values()),
            'categories': {k: len(v) for k, v in edges.items()},
            'bond_types': defaultdict(int),
        }
        
        for category, edge_list in edges.items():
            for edge in edge_list:
                stats['bond_types'][edge['bond_type_name']] += 1
        
        return dict(stats)
    
    def print_statistics(self, edges: Dict[str, List[Dict]]):
        """Print formatted edge statistics."""
        stats = self.get_edge_statistics(edges)
        
        print(f"\n{'='*60}")
        print("EDGE GRAPH STATISTICS")
        print(f"{'='*60}")
        
        print(f"\nTotal edges: {stats['total_edges']}")
        
        print(f"\nBy category:")
        for category, count in sorted(stats['categories'].items()):
            if count > 0:
                print(f"  {category:25s}: {count:6d}")
        
        print(f"\nBy bond type:")
        for bond_type, count in sorted(stats['bond_types'].items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  {bond_type:25s}: {count:6d}")
                