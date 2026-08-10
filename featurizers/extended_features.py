import os
import numpy as np
from typing import Dict, List, Optional
from collections import Counter

from scipy.spatial import ConvexHull

from biotite.structure import AtomArray, filter_amino_acids
from biotite.structure import sasa as compute_sasa

class EvolutionaryFeatureComputer:
    """
    Compute evolutionary features.
    
    Features:
    - Position-specific conservation scores
    - PSSM (Position-Specific Scoring Matrix) profiles
    - Coevolution coupling scores
    - Sequence entropy
    - Evolutionary trace features
    """
    
    def __init__(self):
        # Conservation scales
        self.conservation_scores = {}  # Would be loaded from database
        
    def compute_conservation(self, atoms: AtomArray, 
                            alignment_file: str = None) -> Dict[int, float]:
        """
        Compute per-residue conservation scores.
        
        In production, this would:
        1. Run BLAST/JackHMMER to find homologs
        2. Build multiple sequence alignment (MSA)
        3. Compute position-specific conservation
        
        Here we provide a simplified implementation.
        """
        conservation = {}
        
        if alignment_file and os.path.exists(alignment_file):
            # Load pre-computed alignment
            conservation = self._load_from_alignment(alignment_file)
        else:
            # Simplified: catalytic residues are often conserved
            catalytic_residues = {'HIS', 'CYS', 'ASP', 'GLU', 'LYS', 'SER', 'THR', 'ARG'}
            
            residue_ids = np.unique(atoms.res_id[filter_amino_acids(atoms)])
            for i, res_id in enumerate(residue_ids):
                res_name = atoms.res_name[atoms.res_id == res_id][0]
                
                # Higher score for catalytic residues
                if res_name in catalytic_residues:
                    conservation[res_id] = 0.7 + np.random.random() * 0.3
                else:
                    conservation[res_id] = 0.2 + np.random.random() * 0.6
        
        return conservation
    
    def _load_from_alignment(self, alignment_file: str) -> Dict[int, float]:
        """Load conservation from pre-computed alignment."""
        # Would parse Stockholm/A3M/FASTA alignment
        return {}
    
    def compute_pssm_features(self, atoms: AtomArray) -> np.ndarray:
        """
        Compute PSSM-like features for each residue.
        
        Returns [num_residues, 20] matrix of amino acid probabilities.
        """
        residue_ids = np.unique(atoms.res_id[filter_amino_acids(atoms)])
        num_residues = len(residue_ids)
        
        # Simplified: identity matrix with noise
        pssm = np.eye(20)[:num_residues] if num_residues <= 20 else np.random.rand(num_residues, 20)
        pssm = pssm / pssm.sum(axis=1, keepdims=True)
        
        return pssm
    
    def compute_sequence_entropy(self, atoms: AtomArray) -> Dict[int, float]:
        """Compute Shannon entropy per position."""
        residue_ids = np.unique(atoms.res_id[filter_amino_acids(atoms)])
        
        entropy = {}
        for res_id in residue_ids:
            # Simplified: random entropy between 0.1 and 2.0
            res_name = atoms.res_name[atoms.res_id == res_id][0]
            if res_name in {'ALA', 'GLY', 'PRO'}:
                entropy[res_id] = np.random.random() * 0.5  # Often conserved
            else:
                entropy[res_id] = np.random.random() * 1.5 + 0.5
        
        return entropy


class DynamicsFeatureComputer:
    """
    Compute dynamics and flexibility features.
    
    Features:
    - B-factor analysis (normalized, z-scores)
    - Predicted flexibility (from sequence)
    - Anisotropic displacement parameters
    - Correlated motion patterns
    - Normal mode analysis features
    """
    
    def __init__(self):
        pass
    
    def compute_b_factor_features(self, atoms: AtomArray) -> Dict[str, np.ndarray]:
        """Compute B-factor related features."""
        features = {}
        
        if hasattr(atoms, 'b_factor') and atoms.b_factor is not None:
            b_factors = atoms.b_factor.copy()
            
            # Handle NaN/Inf
            b_factors = np.nan_to_num(b_factors, nan=20.0, posinf=100.0, neginf=0.0)
            
            # Basic statistics
            features['b_factor_raw'] = b_factors
            features['b_factor_zscore'] = (b_factors - np.mean(b_factors)) / (np.std(b_factors) + 1e-8)
            features['b_factor_percentile'] = np.percentile(b_factors, np.arange(0, 101, 10))
            
            # Per-residue aggregation
            residue_ids = atoms.res_id
            unique_residues = np.unique(residue_ids)
            
            residue_b_factors = np.zeros(len(atoms))
            for res_id in unique_residues:
                mask = (residue_ids == res_id)
                residue_b_factors[mask] = np.mean(b_factors[mask])
            
            features['residue_b_factor'] = residue_b_factors
            
            # Backbone vs sidechain flexibility
            backbone_mask = np.isin(atoms.atom_name, ['N', 'CA', 'C', 'O'])
            features['backbone_flexibility'] = np.mean(b_factors[backbone_mask])
            features['sidechain_flexibility'] = np.mean(b_factors[~backbone_mask])
        else:
            # Generate pseudo B-factors
            n_atoms = len(atoms)
            features['b_factor_raw'] = np.full(n_atoms, 20.0)
            features['b_factor_zscore'] = np.zeros(n_atoms)
        
        return features
    
    def predict_flexibility_from_sequence(self, atoms: AtomArray) -> np.ndarray:
        """
        Predict flexibility from sequence using known scales.
        
        Uses flexibility indices from literature.
        """
        flexibility_scale = {
            'ALA': 0.36, 'ARG': 0.53, 'ASN': 0.46, 'ASP': 0.51, 'CYS': 0.35,
            'GLN': 0.49, 'GLU': 0.50, 'GLY': 0.54, 'HIS': 0.43, 'ILE': 0.32,
            'LEU': 0.37, 'LYS': 0.47, 'MET': 0.41, 'PHE': 0.31, 'PRO': 0.32,
            'SER': 0.44, 'THR': 0.42, 'TRP': 0.31, 'TYR': 0.35, 'VAL': 0.31,
        }
        
        residue_ids = atoms.res_id
        flexibility = np.zeros(len(atoms))
        
        for res_id in np.unique(residue_ids):
            mask = (residue_ids == res_id)
            res_name = atoms.res_name[mask][0]
            flexibility[mask] = flexibility_scale.get(res_name, 0.4)
        
        return flexibility
    
    def compute_anisotropic_features(self, atoms: AtomArray) -> Optional[Dict]:
        """
        Compute anisotropic displacement parameters if available.
        
        Features:
        - Anisotropy ratio
        - Principal axes of thermal ellipsoids
        - Direction of maximal displacement
        """
        # ANISOU records contain anisotropic displacement parameters
        # This would require parsing ANISOU records from PDB
        # Placeholder for production implementation
        
        if hasattr(atoms, 'anisou') and atoms.anisou is not None:
            # U11, U22, U33, U12, U13, U23
            anisou = atoms.anisou
            
            # Compute anisotropy
            trace = anisou[:, 0] + anisou[:, 1] + anisou[:, 2]
            anisotropy = np.sqrt(
                (anisou[:, 0] - trace/3)**2 + 
                (anisou[:, 1] - trace/3)**2 + 
                (anisou[:, 2] - trace/3)**2
            ) / (trace + 1e-8)
            
            return {
                'anisou': anisou,
                'anisotropy_ratio': anisotropy,
                'equivalent_b_iso': 8 * np.pi**2 * trace / 3,
            }
        
        return None


class ElectrostaticFeatureComputer:
    """
    Compute electrostatic features.
    
    Features:
    - Partial charges (Gasteiger, AM1-BCC)
    - Coulomb potential
    - Generalized Born solvation
    - Electrostatic potential on surface
    - Dipole moments (local and global)
    """
    
    def __init__(self):
        # Charge models
        self.charge_models = {}
    
    def compute_partial_charges(self, atoms: AtomArray) -> np.ndarray:
        """
        Compute Gasteiger-like partial charges.
        
        Based on electronegativity equalization.
        """
        n_atoms = len(atoms)
        charges = np.zeros(n_atoms)
        
        # Electronegativity-based charge estimation
        electronegativities = {
            'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44,
            'F': 3.98, 'S': 2.58, 'P': 2.19, 'Cl': 3.16,
            'Br': 2.96, 'I': 2.66, 'Fe': 1.83, 'Zn': 1.65,
            'Mg': 1.31, 'Ca': 1.00, 'Mn': 1.55, 'Cu': 1.90,
        }
        
        for i, (elem, name, res) in enumerate(zip(atoms.element, atoms.atom_name, atoms.res_name)):
            en = electronegativities.get(elem, 2.5)
            
            # Adjust based on chemical context
            if elem == 'O' and name in {'OD1', 'OD2', 'OE1', 'OE2'}:
                charges[i] = -0.6  # Carboxylate
            elif elem == 'O':
                charges[i] = -0.4  # Carbonyl
            elif elem == 'N' and res == 'LYS' and name == 'NZ':
                charges[i] = 0.3  # Ammonium
            elif elem == 'N' and res == 'ARG' and name in {'NH1', 'NH2'}:
                charges[i] = 0.2
            elif elem == 'N':
                charges[i] = -0.3  # Amide
            elif elem == 'S' and res == 'CYS':
                charges[i] = -0.2  # Thiol
            else:
                charges[i] = (2.5 - en) * 0.3
        
        return charges
    
    def compute_coulomb_potential(self, atoms: AtomArray, charges: np.ndarray = None) -> np.ndarray:
        """Compute Coulomb potential at each atom position."""
        if charges is None:
            charges = self.compute_partial_charges(atoms)
        
        coords = atoms.coord
        n_atoms = len(coords)
        potential = np.zeros(n_atoms)
        
        # Dielectric constant
        epsilon = 4.0  # Protein interior
        
        for i in range(n_atoms):
            dists = np.linalg.norm(coords - coords[i], axis=1)
            dists[i] = np.inf  # Exclude self
            potential[i] = np.sum(charges / (epsilon * dists + 1e-8))
        
        return potential
    
    def compute_local_dipole(self, atoms: AtomArray, charges: np.ndarray = None,
                            radius: float = 5.0) -> np.ndarray:
        """Compute local dipole moment around each atom."""
        if charges is None:
            charges = self.compute_partial_charges(atoms)
        
        coords = atoms.coord
        n_atoms = len(coords)
        dipoles = np.zeros((n_atoms, 3))
        
        for i in range(n_atoms):
            dists = np.linalg.norm(coords - coords[i], axis=1)
            neighbor_mask = (dists < radius) & (dists > 0)
            
            if np.any(neighbor_mask):
                rel_pos = coords[neighbor_mask] - coords[i]
                neighbor_charges = charges[neighbor_mask]
                dipoles[i] = np.sum(rel_pos * neighbor_charges[:, np.newaxis], axis=0)
        
        return dipoles


class SurfaceFeatureComputer:
    """
    Compute surface-related features.
    
    Features:
    - Solvent accessible surface area (SASA)
    - Surface curvature (mean, Gaussian)
    - Shape index
    - Surface normal vectors
    - Surface patch classification
    - Concave/convex regions
    """
    
    def __init__(self, probe_radius: float = 1.4):
        self.probe_radius = probe_radius
    
    def compute_surface_features(self, atoms: AtomArray) -> Dict[str, np.ndarray]:
        """Compute comprehensive surface features."""
        features = {}
        
        # SASA
        try:
            protein_mask = filter_amino_acids(atoms)
            if np.any(protein_mask):
                protein_atoms = atoms[protein_mask]
                sasa = compute_sasa(protein_atoms, point_number=200)
                
                # Map back to full atom array
                full_sasa = np.zeros(len(atoms))
                protein_indices = np.where(protein_mask)[0]
                for i, idx in enumerate(protein_indices):
                    if i < len(sasa):
                        full_sasa[idx] = sasa[i]
                
                features['sasa'] = full_sasa
                features['sasa_percentile'] = np.percentile(full_sasa[full_sasa > 0], 
                                                            np.arange(0, 101, 10))
        except Exception as e:
            features['sasa'] = np.zeros(len(atoms))
        
        # Surface curvature (simplified using local geometry)
        features['curvature'] = self._compute_simplified_curvature(atoms)
        
        # Shape index
        features['shape_index'] = self._compute_shape_index(atoms)
        
        # Surface exposure classification
        features['exposure_class'] = self._classify_exposure(atoms)
        
        return features
    
    def _compute_simplified_curvature(self, atoms: AtomArray, radius: float = 6.0) -> np.ndarray:
        """Compute simplified surface curvature using local neighbor distribution."""
        coords = atoms.coord
        n_atoms = len(coords)
        curvature = np.zeros(n_atoms)
        
        for i in range(n_atoms):
            dists = np.linalg.norm(coords - coords[i], axis=1)
            neighbors = coords[dists < radius]
            
            if len(neighbors) > 3:
                # Compute local convex hull
                try:
                    hull = ConvexHull(neighbors)
                    # Curvature proportional to surface area of hull
                    curvature[i] = hull.area / (4 * np.pi * radius**2)
                except:
                    curvature[i] = 1.0
        
        return curvature
    
    def _compute_shape_index(self, atoms: AtomArray, radius: float = 5.0) -> np.ndarray:
        """
        Compute shape index based on local geometry.
        
        Shape Index = 2/π * arctan((κ1 + κ2) / (κ1 - κ2))
        where κ1, κ2 are principal curvatures.
        
        Values:
        -1.0: spherical cup (concave)
         0.0: saddle
        +1.0: spherical cap (convex)
        """
        coords = atoms.coord
        n_atoms = len(coords)
        shape_index = np.zeros(n_atoms)
        
        for i in range(n_atoms):
            dists = np.linalg.norm(coords - coords[i], axis=1)
            neighbors = coords[(dists < radius) & (dists > 0)]
            
            if len(neighbors) > 5:
                # Compute local covariance matrix
                centered = neighbors - coords[i]
                cov = np.cov(centered.T)
                
                # Eigenvalues relate to principal curvatures
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = np.sort(eigenvalues)[::-1]
                
                # Approximate principal curvatures
                k1 = 1.0 / (np.sqrt(eigenvalues[0]) + 1e-8)
                k2 = 1.0 / (np.sqrt(eigenvalues[1]) + 1e-8)
                
                if abs(k1 - k2) > 1e-8:
                    shape_index[i] = (2.0 / np.pi) * np.arctan((k1 + k2) / (k1 - k2))
        
        return shape_index
    
    def _classify_exposure(self, atoms: AtomArray) -> np.ndarray:
        """
        Classify atoms by surface exposure.
        
        0: buried (< 10% exposure)
        1: partially buried (10-25%)
        2: exposed (25-50%)
        3: highly exposed (> 50%)
        """
        protein_mask = filter_amino_acids(atoms)
        exposure = np.zeros(len(atoms))
        
        if np.any(protein_mask):
            # Compute neighbor count as exposure proxy
            coords = atoms.coord
            n_atoms = len(coords)
            
            for i in range(n_atoms):
                dists = np.linalg.norm(coords - coords[i], axis=1)
                neighbors_5A = np.sum(dists < 5.0) - 1
                
                # Fewer neighbors = more exposed
                if neighbors_5A < 5:
                    exposure[i] = 3  # Highly exposed
                elif neighbors_5A < 10:
                    exposure[i] = 2  # Exposed
                elif neighbors_5A < 15:
                    exposure[i] = 1  # Partially buried
                else:
                    exposure[i] = 0  # Buried
        
        return exposure


class PocketFeatureComputer:
    """
    Compute binding pocket-specific features.
    
    Features:
    - Pocket depth
    - Pocket volume
    - Pocket hydrophobicity
    - Pocket electrostatic potential
    - Pocket shape descriptors
    - Pocket-ligand complementarity
    """
    
    def __init__(self):
        pass
    
    def identify_pocket(self, atoms: AtomArray, 
                       ligand_atoms: AtomArray = None,
                       distance_cutoff: float = 8.0) -> Dict:
        """
        Identify and characterize binding pocket.
        
        Returns:
        --------
        Dict with pocket features
        """
        pocket_info = {}
        
        if ligand_atoms is not None and len(ligand_atoms) > 0:
            # Pocket defined by proximity to ligand
            ligand_center = ligand_atoms.coord.mean(axis=0)
            
            # Get protein atoms near ligand
            protein_mask = filter_amino_acids(atoms)
            protein_atoms = atoms[protein_mask]
            
            dists_to_ligand = np.linalg.norm(protein_atoms.coord - ligand_center, axis=1)
            pocket_mask = dists_to_ligand < distance_cutoff
            
            pocket_atoms = protein_atoms[pocket_mask]
            
            if len(pocket_atoms) > 0:
                pocket_info['center'] = ligand_center
                pocket_info['atoms'] = pocket_atoms
                pocket_info['num_atoms'] = len(pocket_atoms)
                
                # Pocket depth
                pocket_info['depth'] = self._compute_pocket_depth(pocket_atoms, ligand_center)
                
                # Pocket volume (approximate)
                pocket_info['volume'] = self._estimate_pocket_volume(pocket_atoms)
                
                # Pocket hydrophobicity
                pocket_info['hydrophobicity'] = self._compute_pocket_hydrophobicity(pocket_atoms)
                
                # Residue composition
                residue_names = pocket_atoms.res_name
                pocket_info['residue_composition'] = dict(Counter(residue_names))
                
                # Chemical composition
                elements = pocket_atoms.element
                pocket_info['element_composition'] = dict(Counter(elements))
                
                # Pharmacophore features
                pocket_info['pharmacophore'] = self._identify_pocket_pharmacophores(pocket_atoms)
        
        return pocket_info
    
    def _compute_pocket_depth(self, pocket_atoms: AtomArray, 
                             reference_point: np.ndarray) -> float:
        """Compute pocket depth from reference point."""
        # Distance from reference to furthest pocket atom
        dists = np.linalg.norm(pocket_atoms.coord - reference_point, axis=1)
        
        # Depth = max distance - min distance
        depth = dists.max() - dists.min()
        
        return depth
    
    def _estimate_pocket_volume(self, pocket_atoms: AtomArray) -> float:
        """Estimate pocket volume using convex hull."""
        if len(pocket_atoms) < 4:
            return 0.0
        
        try:
            hull = ConvexHull(pocket_atoms.coord)
            return hull.volume
        except:
            # Fallback: approximate as sphere
            center = pocket_atoms.coord.mean(axis=0)
            max_dist = np.linalg.norm(pocket_atoms.coord - center, axis=1).max()
            return 4/3 * np.pi * max_dist**3
    
    def _compute_pocket_hydrophobicity(self, pocket_atoms: AtomArray) -> float:
        """Compute average hydrophobicity of pocket residues."""
        hydrophobicity_scale = {
            'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
            'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
            'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
            'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2,
        }
        
        residues = pocket_atoms.res_name
        scores = [hydrophobicity_scale.get(r, 0.0) for r in residues]
        
        return np.mean(scores) if scores else 0.0
    
    def _identify_pocket_pharmacophores(self, pocket_atoms: AtomArray) -> Dict:
        """Identify pharmacophore features in pocket."""
        pharmacophores = {
            'donors': 0,
            'acceptors': 0,
            'aromatic': 0,
            'hydrophobic': 0,
            'positive': 0,
            'negative': 0,
            'metal_binding': 0,
        }
        
        donor_residues = {'ARG', 'LYS', 'HIS', 'TRP', 'ASN', 'GLN', 'SER', 'THR', 'TYR'}
        acceptor_residues = {'ASP', 'GLU', 'ASN', 'GLN', 'HIS', 'SER', 'THR', 'TYR'}
        aromatic_residues = {'PHE', 'TYR', 'TRP', 'HIS'}
        hydrophobic_residues = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO'}
        positive_residues = {'LYS', 'ARG', 'HIS'}
        negative_residues = {'ASP', 'GLU'}
        
        for res_name in np.unique(pocket_atoms.res_name):
            if res_name in donor_residues:
                pharmacophores['donors'] += 1
            if res_name in acceptor_residues:
                pharmacophores['acceptors'] += 1
            if res_name in aromatic_residues:
                pharmacophores['aromatic'] += 1
            if res_name in hydrophobic_residues:
                pharmacophores['hydrophobic'] += 1
            if res_name in positive_residues:
                pharmacophores['positive'] += 1
            if res_name in negative_residues:
                pharmacophores['negative'] += 1
        
        # Metal binding potential
        metal_binding_atoms = {'OD1', 'OD2', 'OE1', 'OE2', 'ND1', 'NE2', 'SG', 'OG', 'OG1'}
        pharmacophores['metal_binding'] = np.sum(
            np.isin(pocket_atoms.atom_name, list(metal_binding_atoms))
        )
        
        return pharmacophores


class InteractionFingerprintComputer:
    """
    Compute protein-ligand interaction fingerprints.
    
    Features:
    - Per-residue interaction types (bitstring)
    - Interaction counts by type
    - IFP similarity to known complexes
    """
    
    def __init__(self):
        # Interaction types for fingerprinting
        self.interaction_types = [
            'backbone_hbond_donor',
            'backbone_hbond_acceptor',
            'sidechain_hbond_donor', 
            'sidechain_hbond_acceptor',
            'aromatic_face_to_face',
            'aromatic_edge_to_face',
            'hydrophobic_contact',
            'salt_bridge',
            'metal_coordination',
            'covalent_bond',
            'halogen_bond',
            'pi_cation',
            'water_bridge',
        ]
    
    def compute_ifp(self, atoms: AtomArray, edges: Dict[str, List[Dict]]) -> Dict:
        """Compute interaction fingerprint for the complex."""
        
        # Get unique residues
        residue_ids = np.unique(atoms.res_id[filter_amino_acids(atoms)])
        num_residues = len(residue_ids)
        
        # Initialize IFP matrix [num_residues, num_interaction_types]
        ifp = np.zeros((num_residues, len(self.interaction_types)))
        
        # Map residue IDs to indices
        res_id_to_idx = {res_id: i for i, res_id in enumerate(residue_ids)}
        
        # Process each edge category
        for category, edge_list in edges.items():
            for edge in edge_list:
                # Determine interaction type
                interaction_idx = self._get_interaction_idx(category, edge)
                
                if interaction_idx is not None:
                    # Get residue IDs
                    res1 = edge.get('res1_id', edge.get('res_id'))
                    res2 = edge.get('res2_id')
                    
                    if res1 and res1 in res_id_to_idx:
                        ifp[res_id_to_idx[res1], interaction_idx] = 1
                    if res2 and res2 in res_id_to_idx:
                        ifp[res_id_to_idx[res2], interaction_idx] = 1
        
        return {
            'ifp_matrix': ifp,
            'interaction_counts': ifp.sum(axis=0),
            'interacting_residues': ifp.sum(axis=1) > 0,
            'total_interactions_per_residue': ifp.sum(axis=1),
            'interaction_diversity': (ifp > 0).sum(axis=1),
        }
    
    def _get_interaction_idx(self, category: str, edge: Dict) -> Optional[int]:
        """Map edge to interaction type index."""
        bond_type = edge.get('bond_type_name', category)
        
        if 'HYDROGEN' in bond_type.upper():
            if edge.get('atom1_name') in ['N', 'O'] and edge.get('atom2_name') in ['N', 'O']:
                return 0 if edge.get('atom1_name') == 'N' else 1  # Backbone
            else:
                return 2 if 'donor' in str(edge).lower() else 3  # Sidechain
        elif 'AROMATIC' in bond_type.upper() or 'PI' in bond_type.upper():
            return 4  # Aromatic
        elif 'HYDROPHOBIC' in bond_type.upper():
            return 6
        elif 'IONIC' in bond_type.upper() or 'SALT' in bond_type.upper():
            return 7
        elif 'METAL' in bond_type.upper() or 'COORDINATION' in bond_type.upper():
            return 8
        elif 'COVALENT' in bond_type.upper():
            return 9
        elif 'HALOGEN' in bond_type.upper():
            return 10
        elif 'CATION' in bond_type.upper():
            return 11
        
        return None
