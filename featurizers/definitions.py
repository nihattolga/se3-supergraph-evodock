# ============================================================================
# FEATURE ORGANIZATION
# ============================================================================

from dataclasses import dataclass
from enum import IntEnum
from typing import List

import torch


class NodeFeatureGroup(IntEnum):
    """Organized groups of node features for selective extraction."""
    ATOMIC_IDENTITY = 1           # Z, element, symbol, metal, halogen
    ELECTRONIC = 2                # Charges, electronegativity, ionization
    QUANTUM = 3                   # Hybridization, valence, electrons
    GEOMETRIC = 4                 # Coordinates, distances, density
    STRUCTURAL = 5                # Backbone, sidechain, secondary structure
    RESIDUE_CONTEXT = 6           # Residue type, chain, position0
    PHARMACOPHORIC = 7            # H-bond, aromatic, hydrophobic, charged
    TOPOLOGICAL = 8               # Degree, neighbors, ring membership
    SOLVENT = 9                   # SASA, B-factor, burial
    COARSE_GRAINED = 10           # Bead type, coarse representation
    LEARNABLE = 11                # Trainable embeddings
    STEREOCHEMISTRY = 12          # Chirality, R/S, E/Z
    FUNCTIONAL_GROUP = 13         # Chemical functional groups
    EVOLUTIONARY = 14             # Conservation, PSSM, coevolution
    DYNAMICS = 15                 # B-factor correlations, flexibility
    ELECTROSTATIC = 16            # Poisson-Boltzmann features
    SURFACE = 17                  # Curvature, shape descriptors
    POCKET = 18                   # Pocket-specific features
    CRYSTALLOGRAPHIC = 19         # Resolution, R-factor, occupancy patterns
    ENSEMBLE = 20                 # Multi-model variability
    INTERACTION_FINGERPRINT = 21  # IFP bits
    PHARMACOPHORE_POINT = 22      # Pharmacophore type
    SEQUENCE_PATTERN = 23         # Sequence motifs
    PACKING = 24                  # Packing density, contacts
    BINDING_SITE = 25             # Binding site similarity


class EdgeFeatureGroup(IntEnum):
    """Organized groups of edge features."""
    BOND_IDENTITY = 1             # Bond type, order, conjugated
    GEOMETRIC = 2                 # Distance, direction, angles
    ENERGETIC = 3                 # H-bond, VdW, electrostatic energies
    INTERACTION = 4               # Salt bridge, pi-stack, hydrophobic
    TOPOLOGICAL = 5               # Ring membership, shortest path
    STEREOCHEMISTRY = 6           # E/Z, cis/trans
    RADIAL = 7                    # RBF expansion, distance encodings
    VECTOR = 8                    # Direction vectors, relative positions
    COEVOLUTION = 9               # Evolutionary coupling scores
    DYNAMIC_CORRELATION = 10      # Correlated motions
    ELECTROSTATIC_SCREEN = 11     # Screened electrostatic
    SHAPE_COMPLEMENT = 12         # Shape complementarity
    PHARMACOPHORE_MATCH = 13      # Pharmacophore feature matching
    INTERACTION_FINGERPRINT = 14  # Per-interaction IFP
    CONSERVATION_PAIR = 15        # Co-conservation patterns
    WATER_MEDIATED = 16           # Water-mediated contacts
    ENSEMBLE_CONSISTENCY = 17     # Consistent across models


class BondType(IntEnum):
    """
    Comprehensive chemical bond type enumeration for protein-ligand complexes.
    Covers covalent bonds, non-covalent interactions, and specialized bond types.
    """
    # Fundamental covalent bonds
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    QUADRUPLE = 4

    # Aromatic bond types
    AROMATIC_SINGLE = 5
    AROMATIC_DOUBLE = 6
    AROMATIC_TRIPLE = 7

    # Special coordination and complex bonds
    COORDINATION = 8
    PARTIAL_DOUBLE = 9
    CONJUGATED = 10
    PEPTIDE = 11
    DISULFIDE = 12

    # Non-covalent interactions
    HYDROGEN = 13
    IONIC = 14
    VAN_DER_WAALS = 15
    PI_STACKING = 16
    CATION_PI = 17
    HALOGEN = 18
    METAL_COORDINATION = 19
    SALT_BRIDGE = 20
    HYDROPHOBIC = 21

    # Additional specialized bonds
    COORDINATE_COVALENT = 22
    BACKBONE = 23
    SIDECHAIN = 24
    CROSSLINK = 25
    THIOETHER = 26

    # Ligand-specific bonds
    LIGAND_COVALENT = 27      # Covalent bond between ligand and protein
    LIGAND_COORDINATION = 28  # Metal-ligand coordination bond
    LIGAND_HYDROGEN = 29      # H-bond involving ligand
    LIGAND_HYDROPHOBIC = 30   # Hydrophobic contact with ligand
    LIGAND_PI_STACKING = 31   # π-π stacking with ligand
    LIGAND_IONIC = 32         # Ionic interaction with ligand
    LIGAND_HALOGEN = 33       # Halogen bond with ligand

    @property
    def is_covalent(self) -> bool:
        return self.value in [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 22, 23, 24, 25, 26, 27]

    @property
    def is_non_covalent(self) -> bool:
        return self.value in [13, 14, 15, 16, 17, 18, 20, 21, 28, 29, 30, 31, 32, 33]

    @property
    def is_aromatic(self) -> bool:
        return self.value in [5, 6, 7]

    @property
    def involves_ligand(self) -> bool:
        """Check if bond type involves ligand atoms."""
        return self.value in [27, 28, 29, 30, 31, 32, 33]

    @property
    def bond_order(self) -> float:
        order_map = {
            1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0,
            5: 1.5, 6: 1.5, 7: 1.5,
            9: 1.5, 10: 1.5, 11: 1.33, 22: 1.0,
            27: 1.0,  # Ligand covalent typically single
        }
        return order_map.get(self.value, 0.0)


@dataclass
class BondProperties:
    """Physical and chemical properties associated with bond types."""
    bond_type: BondType
    typical_length: float      # Average bond length in Angstroms (Å)
    strength: float            # Approximate bond dissociation energy in kJ/mol
    is_rotatable: bool
    description: str


@dataclass
class FullNodeFeatures:
    """Complete node feature set."""
    
    # === ATOMIC IDENTITY (8 features) ===
    atomic_number: float           # Z
    element_type: torch.Tensor     # One-hot [num_elements]
    atom_symbol_embed: int         # Learnable embedding ID
    is_metal: bool
    is_halogen: bool
    is_chalcogen: bool
    is_noble_gas: bool
    
    # === ELECTRONIC (7 features) ===
    formal_charge: float
    partial_charge: float
    electronegativity: float
    polarizability: float
    valence_electrons: int
    ionization_potential: float
    electron_affinity: float
    
    # === QUANTUM/BONDING (8 features) ===
    hybridization: int             # 0:unknown, 1:sp, 2:sp2, 3:sp3, 4:aromatic
    is_sp2_planar: bool
    is_sp_linear: bool
    num_bonds: int                 # Degree
    num_heavy_atom_bonds: int
    num_hydrogens: int
    implicit_valence: int
    residual_valence: float        # ideal - current
    
    # === STEREOCHEMISTRY (4 features) ===
    stereochemistry_r_s: int       # 0:none, 1:R, 2:S
    chirality_tag: int             # 0:none, 1:tetrahedral, 2:other
    cis_trans: int                 # 0:none, 1:cis, 2:trans
    stereochemistry_vector: torch.Tensor  # [4] one-hot
    
    # === STRUCTURAL (6 features) ===
    is_backbone: bool
    is_sidechain: bool
    is_terminal_sidechain: bool
    secondary_structure: int       # 0:coil, 1:helix, 2:sheet, 3:turn
    is_in_ring: bool
    ring_size: int
    
    # === RESIDUE CONTEXT (24 features) ===
    residue_type: int              # 0-19: standard AA, 20+: others
    residue_type_onehot: torch.Tensor  # [22] one-hot (20 AA + ligand + other)
    residue_id: int
    chain_id: str
    chain_id_embed: int
    trans_chain_position: float    # Normalized position in chain
    residue_name: str
    
    # === FUNCTIONAL GROUP (12 features) ===
    functional_group_type: int     # 0-11: predefined groups
    functional_group_onehot: torch.Tensor  # [12] one-hot
    is_amide: bool
    is_carboxyl: bool
    is_hydroxyl: bool
    is_amine: bool
    is_thiol: bool
    is_phosphate: bool
    is_aromatic_ring_atom: bool
    
    # === PHARMACOPHORIC (10 features) ===
    is_hbond_donor: bool
    is_hbond_acceptor: bool
    donor_strength: float
    acceptor_strength: float
    is_aromatic: bool
    is_pi_system: bool
    is_cation: bool
    is_anion: bool
    is_hydrophobic: bool
    hydrophobicity_score: float
    metal_binding_atom: bool
    
    # === GEOMETRIC (12 features) ===
    coord_x: float
    coord_y: float
    coord_z: float
    relative_position_to_centroid: torch.Tensor  # [3]
    distance_to_ligand_min: float
    distance_to_pocket_center: float
    b_factor: float
    occupancy: float
    local_atom_density_4A: float
    local_atom_density_6A: float
    local_atom_density_8A: float
    
    # === SOLVENT (3 features) ===
    solvent_accessibility: float   # SASA
    relative_sasa: float
    burial_depth: float
    
    # === TOPOLOGICAL (5 features) ===
    degree: int
    clustering_coefficient: float
    betweenness_centrality: float
    num_neighbors: int
    
    # === COARSE-GRAINED (2 features) ===
    bead_type: int                 # Coarse-grained bead type
    coarse_embed_id: int
    
    # === LEARNABLE (1 feature) ===
    learnable_embedding_id: int
    
    # === LEGACY/CHEMINFORMATICS (5 features) ===
    atom_name_pdb: str
    atom_type_mmff: int            # MMFF atom type
    atom_type_gaff: int            # GAFF atom type
    covalent_affinity: float
    covalent_radius: float
    
    def to_vector(self, groups: List[NodeFeatureGroup] = None) -> torch.Tensor:
        """Convert to flat feature vector."""
        if groups is None:
            groups = list(NodeFeatureGroup)
        
        vectors = []
        
        if NodeFeatureGroup.ATOMIC_IDENTITY in groups:
            vectors.append(torch.tensor([
                self.atomic_number / 100,
                float(self.is_metal),
                float(self.is_halogen),
                float(self.is_chalcogen),
                float(self.is_noble_gas),
            ]))
            vectors.append(self.element_type)
        
        if NodeFeatureGroup.ELECTRONIC in groups:
            vectors.append(torch.tensor([
                self.formal_charge,
                self.partial_charge,
                self.electronegativity / 4,
                self.polarizability / 50,
                self.valence_electrons / 12,
                self.ionization_potential / 25,
                self.electron_affinity / 4,
            ]))
        
        if NodeFeatureGroup.QUANTUM in groups:
            vectors.append(torch.tensor([
                self.hybridization / 4,
                float(self.is_sp2_planar),
                float(self.is_sp_linear),
                self.num_bonds / 8,
                self.num_heavy_atom_bonds / 8,
                self.num_hydrogens / 4,
                self.implicit_valence / 8,
                self.residual_valence,
            ]))
        
        if NodeFeatureGroup.STEREOCHEMISTRY in groups:
            vectors.append(torch.tensor([
                self.stereochemistry_r_s / 2,
                self.chirality_tag / 2,
                self.cis_trans / 2,
            ]))
            vectors.append(self.stereochemistry_vector)
        
        if NodeFeatureGroup.STRUCTURAL in groups:
            vectors.append(torch.tensor([
                float(self.is_backbone),
                float(self.is_sidechain),
                float(self.is_terminal_sidechain),
                self.secondary_structure / 3,
                float(self.is_in_ring),
                self.ring_size / 10,
            ]))
        
        if NodeFeatureGroup.RESIDUE_CONTEXT in groups:
            vectors.append(self.residue_type_onehot)
            vectors.append(torch.tensor([
                self.trans_chain_position,
            ]))
        
        if NodeFeatureGroup.FUNCTIONAL_GROUP in groups:
            vectors.append(self.functional_group_onehot)
        
        if NodeFeatureGroup.PHARMACOPHORIC in groups:
            vectors.append(torch.tensor([
                float(self.is_hbond_donor),
                float(self.is_hbond_acceptor),
                self.donor_strength,
                self.acceptor_strength,
                float(self.is_aromatic),
                float(self.is_pi_system),
                float(self.is_cation),
                float(self.is_anion),
                float(self.is_hydrophobic),
                self.hydrophobicity_score / 5,
                float(self.metal_binding_atom),
            ]))
        
        if NodeFeatureGroup.GEOMETRIC in groups:
            vectors.append(torch.tensor([
                self.coord_x / 100,
                self.coord_y / 100,
                self.coord_z / 100,
                self.distance_to_ligand_min / 50,
                self.distance_to_pocket_center / 50,
                self.b_factor / 100,
                self.occupancy,
                self.local_atom_density_4A / 30,
                self.local_atom_density_6A / 80,
                self.local_atom_density_8A / 150,
            ]))
            vectors.append(self.relative_position_to_centroid)
        
        if NodeFeatureGroup.SOLVENT in groups:
            vectors.append(torch.tensor([
                self.solvent_accessibility / 200,
                self.relative_sasa,
                self.burial_depth / 30,
            ]))
        
        if NodeFeatureGroup.TOPOLOGICAL in groups:
            vectors.append(torch.tensor([
                self.degree / 12,
                self.clustering_coefficient,
                self.betweenness_centrality,
                self.num_neighbors / 20,
            ]))
        
        if NodeFeatureGroup.COARSE_GRAINED in groups:
            vectors.append(torch.tensor([
                self.bead_type / 20,
                self.coarse_embed_id / 100,
            ]))
        
        return torch.cat(vectors) if vectors else torch.tensor([])

@dataclass
class FullEdgeFeatures:
    """Complete edge feature set."""
    
    # === BOND IDENTITY (6 features) ===
    bond_type: int                # 0-11: bond type index
    bond_type_onehot: torch.Tensor  # [12] one-hot
    is_conjugated: bool
    is_aromatic_bond: bool
    bond_order: float
    cis_trans_e_z: int            # 0:none, 1:cis, 2:trans
    
    # === GEOMETRIC (10 features) ===
    distance: float
    distance_inverse: float       # 1/distance
    distance_rbf: torch.Tensor    # [20] RBF expansion
    unit_direction_vector: torch.Tensor  # [3]
    relative_position_vector: torch.Tensor  # [3]
    bond_angle: float             # i-j-k angle
    dihedral_angle: float         # i-j-k-l dihedral
    cosine_distance: float
    spherical_angles: torch.Tensor  # [2] theta, phi
    
    # === ENERGETIC (5 features) ===
    hbond_energy: float
    vdw_energy: float
    electrostatic_energy: float
    total_interaction_energy: float
    
    # === INTERACTION (8 features) ===
    is_hydrogen_bond: bool
    is_salt_bridge: bool
    is_pi_pi_stacking: bool
    is_cation_pi: bool
    is_hydrophobic_contact: bool
    is_halogen_bond: bool
    is_metal_coordination: bool
    is_contact: bool              # Within cutoff
    
    # === TOPOLOGICAL (5 features) ===
    topological_distance: int     # Shortest path length
    in_same_ring: bool
    ring_membership: bool
    ring_size: int
    
    def to_vector(self, groups: List[EdgeFeatureGroup] = None) -> torch.Tensor:
        """Convert to feature vector."""
        if groups is None:
            groups = list(EdgeFeatureGroup)
        
        vectors = []
        
        if EdgeFeatureGroup.BOND_IDENTITY in groups:
            vectors.append(self.bond_type_onehot)
            vectors.append(torch.tensor([
                float(self.is_conjugated),
                float(self.is_aromatic_bond),
                self.bond_order / 3,
                self.cis_trans_e_z / 2,
            ]))
        
        if EdgeFeatureGroup.GEOMETRIC in groups:
            vectors.append(torch.tensor([
                self.distance / 10,
                self.distance_inverse * 10,
                self.bond_angle / 180,
                self.dihedral_angle / 180,
                self.cosine_distance,
            ]))
            vectors.append(self.unit_direction_vector)
        
        if EdgeFeatureGroup.RADIAL in groups:
            vectors.append(self.distance_rbf)
        
        if EdgeFeatureGroup.ENERGETIC in groups:
            vectors.append(torch.tensor([
                self.hbond_energy / 50,
                self.vdw_energy / 10,
                self.electrostatic_energy / 100,
                self.total_interaction_energy / 100,
            ]))
        
        if EdgeFeatureGroup.INTERACTION in groups:
            vectors.append(torch.tensor([
                float(self.is_hydrogen_bond),
                float(self.is_salt_bridge),
                float(self.is_pi_pi_stacking),
                float(self.is_cation_pi),
                float(self.is_hydrophobic_contact),
                float(self.is_halogen_bond),
                float(self.is_metal_coordination),
                float(self.is_contact),
            ]))
        
        if EdgeFeatureGroup.TOPOLOGICAL in groups:
            vectors.append(torch.tensor([
                self.topological_distance / 20,
                float(self.in_same_ring),
                float(self.ring_membership),
                self.ring_size / 10,
            ]))
        
        return torch.cat(vectors) if vectors else torch.tensor([])
