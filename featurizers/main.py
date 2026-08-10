import torch
import json
from pathlib import Path
from typing import Dict
import numpy as np

from .pdb_processor import PDBProcessor
from .extended_features import (
    EvolutionaryFeatureComputer, DynamicsFeatureComputer, ElectrostaticFeatureComputer,
    SurfaceFeatureComputer, PocketFeatureComputer, InteractionFingerprintComputer
)
from .node_features import FullNodeFeatureExtractor
from .edge_features import FullEdgeFeatureExtractor

class CompletePDFeaturizer:
    """
    Complete featurizer that takes a PDB file and extracts ALL features.
    
    This is the main class you should use.
    """
    
    def __init__(self, pdb_path: str, device: str = 'cpu'):
        self.pdb_path = pdb_path
        self.device = device
        
        print(f"\n{'='*70}")
        print(f"Processing PDB: {pdb_path}")
        print(f"{'='*70}")
        
        # Load and process PDB
        print("\n1. Loading structure...")
        self.processor = PDBProcessor(pdb_path)
        self.atoms = self.processor.atoms
        
        # Categorize residues
        print("\n2. Classifying residues...")
        self.residue_categories = self.processor.categorize_residues()
        
        # Build edges
        print("\n3. Building molecular graph...")
        self.edges = self.processor.build_edges()
        
        # Initialize feature computers
        print("\n4. Initializing feature computers...")
        self.evolutionary = EvolutionaryFeatureComputer()
        self.dynamics = DynamicsFeatureComputer()
        self.electrostatic = ElectrostaticFeatureComputer()
        self.surface = SurfaceFeatureComputer()
        self.pocket = PocketFeatureComputer()
        self.ifp_computer = InteractionFingerprintComputer()
        
        # Initialize base feature extractors
        self.node_extractor = FullNodeFeatureExtractor(device=device)
        self.edge_extractor = FullEdgeFeatureExtractor(device=device)
        
        # Cache for computed features
        self._feature_cache = {}
        
        # Extract all features
        self._extract_all_features()
        
        print(f"\n{'='*70}")
        print("Featurization complete!")
        print(f"{'='*70}")
    
    def _extract_all_features(self):
        """Extract all features from the PDB file."""
        
        # Base node features
        print("\n5. Extracting base node features...")
        self.base_node_features = self.node_extractor.extract_from_biotite(
            self.atoms, self.edges
        )
        
        # Base edge features
        print("6. Extracting base edge features...")
        self.base_edge_features = self.edge_extractor.extract_from_edges(
            self.atoms, self.edges
        )
        
        # Additional features
        print("7. Computing additional features...")
        
        # Evolutionary features
        print("  - Evolutionary conservation...")
        self.conservation = self.evolutionary.compute_conservation(self.atoms)
        self.pssm = self.evolutionary.compute_pssm_features(self.atoms)
        
        # Dynamics features
        print("  - Dynamics and flexibility...")
        self.b_factor_features = self.dynamics.compute_b_factor_features(self.atoms)
        self.flexibility = self.dynamics.predict_flexibility_from_sequence(self.atoms)
        
        # Electrostatic features
        print("  - Electrostatic properties...")
        self.charges = self.electrostatic.compute_partial_charges(self.atoms)
        self.coulomb = self.electrostatic.compute_coulomb_potential(self.atoms, self.charges)
        self.dipoles = self.electrostatic.compute_local_dipole(self.atoms, self.charges)
        
        # Surface features
        print("  - Surface properties...")
        self.surface_features = self.surface.compute_surface_features(self.atoms)
        
        # Pocket features
        print("  - Binding pocket analysis...")
        ligand_atoms = self.processor.get_ligand_atoms()
        self.pocket_info = self.pocket.identify_pocket(self.atoms, ligand_atoms)
        
        # Interaction fingerprints
        print("  - Interaction fingerprints...")
        self.ifp = self.ifp_computer.compute_ifp(self.atoms, self.edges)
        
        print("8. Building final feature matrices...")
        self.node_feature_matrix = self._build_node_feature_matrix()
        self.edge_feature_matrix = self._build_edge_feature_matrix()
    
    def _build_node_feature_matrix(self) -> torch.Tensor:
        """Build complete node feature matrix."""
        n_atoms = len(self.atoms)
        feature_list = []
        
        # Base features from FullNodeFeatureExtractor
        base_vectors = [f.to_vector() for f in self.base_node_features]
        if base_vectors:
            base_matrix = torch.stack(base_vectors)
        else:
            base_matrix = torch.zeros(n_atoms, 0)
        
        # Additional feature columns
        additional_features = []
        
        # Conservation per atom (broadcast from residue)
        cons_per_atom = np.array([
            self.conservation.get(self.atoms.res_id[i], 0.5) 
            for i in range(n_atoms)
        ])
        additional_features.append(torch.tensor(cons_per_atom, dtype=torch.float32).unsqueeze(-1))
        
        # B-factor z-scores
        if 'b_factor_zscore' in self.b_factor_features:
            b_zscore = torch.tensor(self.b_factor_features['b_factor_zscore'], dtype=torch.float32)
            additional_features.append(b_zscore.unsqueeze(-1))
        
        # Flexibility
        flex = torch.tensor(self.flexibility, dtype=torch.float32)
        additional_features.append(flex.unsqueeze(-1))
        
        # Charges
        charges = torch.tensor(self.charges, dtype=torch.float32)
        additional_features.append(charges.unsqueeze(-1))
        
        # Coulomb potential
        coulomb = torch.tensor(self.coulomb, dtype=torch.float32)
        additional_features.append(coulomb.unsqueeze(-1))
        
        # SASA
        if 'sasa' in self.surface_features:
            sasa = torch.tensor(self.surface_features['sasa'] / 200, dtype=torch.float32)
            additional_features.append(sasa.unsqueeze(-1))
        
        # Curvature
        if 'curvature' in self.surface_features:
            curvature = torch.tensor(self.surface_features['curvature'], dtype=torch.float32)
            additional_features.append(curvature.unsqueeze(-1))
        
        # Shape index
        if 'shape_index' in self.surface_features:
            shape = torch.tensor(self.surface_features['shape_index'], dtype=torch.float32)
            additional_features.append(shape.unsqueeze(-1))
        
        # Exposure class
        if 'exposure_class' in self.surface_features:
            exposure = torch.tensor(self.surface_features['exposure_class'] / 3, dtype=torch.float32)
            additional_features.append(exposure.unsqueeze(-1))
        
        # Concatenate all features
        if additional_features:
            additional_matrix = torch.cat(additional_features, dim=-1)
            final_matrix = torch.cat([base_matrix, additional_matrix], dim=-1)
        else:
            final_matrix = base_matrix
        
        return final_matrix
    
    def _build_edge_feature_matrix(self) -> Dict[str, torch.Tensor]:
        """Build complete edge feature matrices per category."""
        edge_matrices = {}
        
        for category, feature_list in self.base_edge_features.items():
            if feature_list:
                vectors = [f.to_vector() for f in feature_list]
                edge_matrices[category] = torch.stack(vectors)
        
        return edge_matrices
    
    def get_pyg_data(self) -> Dict:
        """
        Get PyTorch Geometric compatible data.
        
        Returns:
        --------
        Dict with:
        - x: Node features
        - edge_index: Edge indices
        - edge_attr: Edge features
        - pos: 3D coordinates
        - All metadata
        """
        # Build edge index from all edges
        edge_indices = []
        edge_attrs = []
        
        for category, edge_list in self.edges.items():
            if category in self.edge_feature_matrix:
                edge_feats = self.edge_feature_matrix[category]
                
                for i, edge in enumerate(edge_list):
                    idx1 = edge.get('atom1_index', 0)
                    idx2 = edge.get('atom2_index', 0)
                    
                    edge_indices.append([idx1, idx2])
                    edge_indices.append([idx2, idx1])
                    
                    if i < len(edge_feats):
                        edge_attrs.append(edge_feats[i])
                        edge_attrs.append(edge_feats[i])
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices).t().contiguous()
            edge_attr = torch.stack(edge_attrs) if edge_attrs else torch.zeros((len(edge_indices), 1))
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1))
        
        return {
            'x': self.node_feature_matrix.to(self.device),
            'edge_index': edge_index.to(self.device),
            'edge_attr': edge_attr.to(self.device),
            'pos': torch.tensor(self.atoms.coord, dtype=torch.float32).to(self.device),
            'num_nodes': len(self.atoms),
            'metadata': self.get_metadata(),
        }
    
    def get_metadata(self) -> Dict:
        """Get all metadata about the featurization."""
        return {
            'pdb_path': self.pdb_path,
            'num_atoms': len(self.atoms),
            'num_residues': len(np.unique(self.atoms.res_id)),
            'num_chains': len(np.unique(self.atoms.chain_id)),
            'node_feature_dim': self.node_feature_matrix.size(-1),
            'num_edges': sum(len(el) for el in self.edges.values()),
            'resolution': self.processor.resolution,
            'r_factor': self.processor.r_factor,
            'residue_categories': {
                cat: len(res_list) 
                for cat, res_list in self.residue_categories.items()
            },
            'pocket_info': {
                k: v for k, v in self.pocket_info.items() 
                if k != 'atoms'
            } if self.pocket_info else None,
            'interaction_fingerprint': {
                k: v for k, v in self.ifp.items() 
                if k != 'ifp_matrix'
            } if self.ifp else None,
        }
    
    def print_summary(self):
        """Print comprehensive featurization summary."""
        metadata = self.get_metadata()
        
        print(f"\n{'='*70}")
        print("FEATURIZATION SUMMARY")
        print(f"{'='*70}")
        
        print(f"\nPDB File: {metadata['pdb_path']}")
        print(f"Resolution: {metadata['resolution'] or 'N/A'} Å")
        print(f"R-factor: {metadata['r_factor'] or 'N/A'}")
        
        print(f"\nStructure:")
        print(f"  Atoms: {metadata['num_atoms']}")
        print(f"  Residues: {metadata['num_residues']}")
        print(f"  Chains: {metadata['num_chains']}")
        
        print(f"\nGraph:")
        print(f"  Node features: {metadata['node_feature_dim']}D")
        print(f"  Total edges: {metadata['num_edges']}")
        
        print(f"\nResidue Classification:")
        for cat, count in metadata['residue_categories'].items():
            if count > 0:
                print(f"  {cat:20s}: {count:5d}")
        
        if metadata['pocket_info']:
            print(f"\nBinding Pocket:")
            for k, v in metadata['pocket_info'].items():
                if isinstance(v, (int, float, str)):
                    print(f"  {k}: {v}")
                elif isinstance(v, dict):
                    print(f"  {k}:")
                    for kk, vv in v.items():
                        print(f"    {kk}: {vv}")
        
        if metadata['interaction_fingerprint']:
            print(f"\nInteractions:")
            ifp = metadata['interaction_fingerprint']
            if 'total_interactions_per_residue' in ifp:
                interacting = np.sum(ifp['total_interactions_per_residue'] > 0)
                print(f"  Interacting residues: {interacting}")
    
    def save_features(self, output_path: str):
        """Save all features to disk."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save node features
        torch.save(self.node_feature_matrix, output_path / 'node_features.pt')
        
        # Save edge features
        torch.save(self.edge_feature_matrix, output_path / 'edge_features.pt')
        
        # Save metadata
        with open(output_path / 'metadata.json', 'w') as f:
            # Convert numpy arrays to lists for JSON
            metadata = self.get_metadata()
            def convert(obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                elif isinstance(obj, (np.floating,)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(i) for i in obj]
                return obj
            
            json.dump(convert(metadata), f, indent=2)
        
        print(f"\nFeatures saved to {output_path}")
