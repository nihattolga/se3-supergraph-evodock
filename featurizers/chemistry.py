from typing import Dict

class ChemicalPeriodicTable:
    """Complete periodic table data for all features."""
    
    # Element data: Z, mass, vdw_radius, covalent_radius, en, ionization, 
    #               electron_affinity, polarizability, valence_e, common_valence
    ELEMENTS = {
        'H':  (1, 1.008, 1.20, 0.31, 2.20, 13.598, 0.754, 0.667, 1, 1),
        'He': (2, 4.003, 1.40, 0.28, 0.00, 24.587, -0.500, 0.205, 0, 0),
        'Li': (3, 6.941, 1.82, 1.28, 0.98, 5.392, 0.618, 24.300, 1, 1),
        'Be': (4, 9.012, 1.53, 0.96, 1.57, 9.323, 0.000, 5.600, 2, 2),
        'B':  (5, 10.811, 1.92, 0.84, 2.04, 8.298, 0.277, 3.030, 3, 3),
        'C':  (6, 12.011, 1.70, 0.76, 2.55, 11.260, 1.263, 1.760, 4, 4),
        'N':  (7, 14.007, 1.55, 0.71, 3.04, 14.534, -0.070, 1.100, 3, 3),
        'O':  (8, 15.999, 1.52, 0.66, 3.44, 13.618, 1.461, 0.802, 2, 2),
        'F':  (9, 18.998, 1.47, 0.57, 3.98, 17.423, 3.401, 0.557, 1, 1),
        'Ne': (10, 20.180, 1.54, 0.58, 0.00, 21.565, -1.200, 0.396, 0, 0),
        'Na': (11, 22.990, 2.27, 1.66, 0.93, 5.139, 0.548, 24.110, 1, 1),
        'Mg': (12, 24.305, 1.73, 1.41, 1.31, 7.646, 0.000, 10.600, 2, 2),
        'Al': (13, 26.982, 1.84, 1.21, 1.61, 5.986, 0.441, 8.340, 3, 3),
        'Si': (14, 28.086, 2.10, 1.11, 1.90, 8.152, 1.385, 5.380, 4, 4),
        'P':  (15, 30.974, 1.80, 1.07, 2.19, 10.487, 0.747, 3.630, 3, 3),
        'S':  (16, 32.065, 1.80, 1.05, 2.58, 10.360, 2.077, 2.900, 2, 2),
        'Cl': (17, 35.453, 1.75, 1.02, 3.16, 12.968, 3.613, 2.180, 1, 1),
        'Ar': (18, 39.948, 1.88, 1.06, 0.00, 15.760, -1.000, 1.641, 0, 0),
        'K':  (19, 39.098, 2.75, 2.03, 0.82, 4.341, 0.501, 43.340, 1, 1),
        'Ca': (20, 40.078, 2.31, 1.76, 1.00, 6.113, 0.018, 22.800, 2, 2),
        'Sc': (21, 44.956, 2.15, 1.70, 1.36, 6.561, 0.188, 17.800, 3, 3),
        'Ti': (22, 47.867, 2.11, 1.60, 1.54, 6.828, 0.079, 14.600, 4, 4),
        'V':  (23, 50.942, 2.07, 1.53, 1.63, 6.746, 0.524, 12.400, 5, 5),
        'Cr': (24, 51.996, 2.06, 1.39, 1.66, 6.767, 0.666, 11.600, 6, 6),
        'Mn': (25, 54.938, 2.05, 1.39, 1.55, 7.434, 0.000, 9.400, 7, 7),
        'Fe': (26, 55.845, 2.04, 1.32, 1.83, 7.902, 0.163, 8.400, 8, 8),
        'Co': (27, 58.933, 2.00, 1.26, 1.88, 7.881, 0.660, 7.500, 9, 9),
        'Ni': (28, 58.693, 1.97, 1.24, 1.91, 7.640, 1.156, 6.800, 10, 10),
        'Cu': (29, 63.546, 1.96, 1.32, 1.90, 7.726, 1.228, 6.100, 11, 11),
        'Zn': (30, 65.380, 2.01, 1.22, 1.65, 9.394, 0.000, 7.100, 12, 12),
        'Br': (35, 79.904, 1.85, 1.20, 2.96, 11.814, 3.365, 3.050, 1, 1),
        'I':  (53, 126.90, 1.98, 1.39, 2.66, 10.451, 3.059, 4.700, 1, 1),
        'Se': (34, 78.960, 1.90, 1.20, 2.55, 9.752, 2.021, 3.770, 2, 2),
    }
    
    # Metal classification
    METALS = {'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 
              'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Rb', 'Sr', 'Y',
              'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
              'Sn', 'Sb', 'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os',
              'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po'}
    
    # Halogen classification
    HALOGENS = {'F', 'Cl', 'Br', 'I', 'At'}
    
    # Chalcogens
    CHALCOGENS = {'O', 'S', 'Se', 'Te', 'Po'}
    
    # Noble gases
    NOBLE_GASES = {'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'}
    
    @classmethod
    def get_element_features(cls, element: str) -> Dict:
        """Get all features for an element."""
        if element not in cls.ELEMENTS:
            element = 'C'  # Default to carbon
        
        z, mass, vdw, cov, en, ion, ea, pol, val, common_val = cls.ELEMENTS[element]
        
        return {
            'atomic_number': z,
            'atomic_mass': mass,
            'vdw_radius': vdw,
            'covalent_radius': cov,
            'electronegativity': en,
            'ionization_potential': ion,
            'electron_affinity': ea,
            'polarizability': pol,
            'valence_electrons': val,
            'common_valence': common_val,
            'is_metal': element in cls.METALS,
            'is_halogen': element in cls.HALOGENS,
            'is_chalcogen': element in cls.CHALCOGENS,
            'is_noble_gas': element in cls.NOBLE_GASES,
        }
