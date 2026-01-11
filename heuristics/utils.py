from typing import Optional, Tuple, Union, List, Dict, Any, Set, Iterable
import logging
import math
import re

from .vectors import VectorsError, calculate_param
import numpy as np

from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import rdMolAlign

try:
    from rmgpy.molecule.molecule import Molecule as RMG_Molecule
    from rmgpy.molecule import converter as rmg_converter
except Exception:
    RMG_Molecule = None
    rmg_converter = None


DEFAULT_COMPARISON_D_TOL = 2.0  # degrees
DEFAULT_CONSOLIDATION_R_TOL = 1e-4
DEFAULT_CONSOLIDATION_A_TOL = 1e-3
DEFAULT_CONSOLIDATION_D_TOL = 1e-3
TOL_180 = 0.9  # degrees
KEY_FROM_LEN = {2: 'R', 3: 'A', 4: 'D'}
logger = logging.getLogger(__name__)


class ZMatError(Exception):
    pass


@dataclass
class MolWithXYZ:
    """
    Minimal replacement for ARCSpecies/Molecule wrapper:
    - mol: RDKit mol
    - xyz: optional xyz dict with 'symbols' and 'coords' aligned to mol atom order
    """
    mol: Chem.Mol
    xyz: Optional[dict] = None
    label: str = "S"

    @property
    def number_of_atoms(self) -> int:
        return int(self.mol.GetNumAtoms())

    def get_xyz(self) -> dict:
        if self.xyz is not None:
            return self.xyz
        # If xyz wasn't provided, read from conformer 0
        conf = self.mol.GetConformer()
        coords = [(float(conf.GetAtomPosition(i).x),
                   float(conf.GetAtomPosition(i).y),
                   float(conf.GetAtomPosition(i).z)) for i in range(self.mol.GetNumAtoms())]
        symbols = tuple(a.GetSymbol() for a in self.mol.GetAtoms())
        return {"symbols": symbols, "coords": tuple(coords)}


def _atomic_symbols(m: Chem.Mol) -> Tuple[str, ...]:
    return tuple(a.GetSymbol() for a in m.GetAtoms())


def _formula_count(m: Chem.Mol) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for a in m.GetAtoms():
        d[a.GetSymbol()] = d.get(a.GetSymbol(), 0) + 1
    return d


def _bond_signature_counts(m: Chem.Mol) -> Dict[Tuple[str, str], int]:
    """
    Bond counts ignoring bond order, keyed by sorted element pair.
    Similar spirit to get_bonds_dict().
    """
    d: Dict[Tuple[str, str], int] = {}
    for b in m.GetBonds():
        e1 = b.GetBeginAtom().GetSymbol()
        e2 = b.GetEndAtom().GetSymbol()
        key = tuple(sorted((e1, e2)))
        d[key] = d.get(key, 0) + 1
    return d


def _check_mols_before_mapping(m1: Chem.Mol, m2: Chem.Mol) -> bool:
    if m1 is None or m2 is None:
        return False
    if m1.GetNumAtoms() == 0 or m2.GetNumAtoms() == 0:
        return False
    if m1.GetNumAtoms() != m2.GetNumAtoms():
        return False
    if _formula_count(m1) != _formula_count(m2):
        return False
    if _bond_signature_counts(m1) != _bond_signature_counts(m2):
        return False
    return True


def _identity_map(n: int, map_type: str) -> Union[List[int], Dict[int, int]]:
    if map_type == "dict":
        return {i: i for i in range(n)}
    return list(range(n))


def _invert_list_map(m12: List[int]) -> List[int]:
    """
    Given mapping i->m12[i], produce inverse mapping j->i.
    Assumes bijection.
    """
    inv = [0] * len(m12)
    for i, j in enumerate(m12):
        inv[j] = i
    return inv

def get_new_map_based_on_zmat_1(zmat_1: dict,
                                zmat_2: dict,
                                reactants_reversed: bool = False,
                                ) -> dict:
    """
    Generate an initial map for the combined zmats, here only consider ``zmat_1``.

    Args:
        zmat_1 (dict): The zmat describing R1H. Contains a dummy atom at the end if a2 is linear.
        zmat_2 (dict): The zmat describing R2H.
        reactants_reversed (bool, optional): Whether the reactants were reversed relative to the RMG template.

    Returns:
        dict: The initial map for the combined zmats.
    """
    new_map = dict()
    val_inc = len(zmat_2['symbols']) - 1 if reactants_reversed else 0
    for key, val in zmat_1['map'].items():
        if isinstance(val, str) and 'X' in val:
            new_map[key] = f'X{int(val[1:]) + val_inc}'
        else:
            new_map[key] = val + val_inc
    return new_map


def rebuild_map(old_map: Dict[int, Union[int, str]], dropped_idx: int) -> Dict[int, Union[int, str]]:
    """
    Rebuild the Z-matrix to XYZ index map after removing atom 0.

    - Z-matrix indices (keys) are shifted down by 1 if > 0
    - XYZ indices (values) are shifted down by 1 if > dropped_idx
    - Dummy atoms ('Xn') are handled appropriately

    Args:
        old_map (dict): Original map from Z-matrix to XYZ indices (int or 'Xn')
        dropped_idx (int): Cartesian index of the removed atom

    Returns:
        dict: New map with updated indices
    """
    new_map = {}
    for old_z_i, old_xyz_i in old_map.items():
        if old_z_i == 0:
            continue  # drop atom 0

        new_z_i = old_z_i - 1

        if isinstance(old_xyz_i, str) and old_xyz_i.startswith('X'):
            x_idx = int(old_xyz_i[1:])
            new_x_idx = x_idx - 1 if x_idx > dropped_idx else x_idx
            new_xyz_i = f'X{new_x_idx}'
        else:
            new_xyz_i = old_xyz_i - 1 if old_xyz_i > dropped_idx else old_xyz_i

        new_map[new_z_i] = new_xyz_i

    return new_map

def remove_zmat_atom_0(zmat: dict) -> dict:
    """
    Remove atom 0 from a Z-matrix complete structure, dropping all references to it
    and renumbering the remaining atoms, parameters, and map accordingly.
    """
    if len(zmat['symbols']) <= 1:
        return {'symbols': (), 'coords': (), 'vars': {}, 'map': {}}
    if len(zmat['symbols']) == 2:
        return {'symbols': (zmat['symbols'][1],), 'coords': ((None, None, None),), 'vars': {}, 'map': {0: 0}}
    orig_map0 = zmat['map'][0]
    dropped_idx = int(orig_map0[1:]) if isinstance(orig_map0, str) and orig_map0.startswith('X') else orig_map0
    purged = purge_references_to_atom_0(zmat)
    dropped = drop_symbol_and_coords_row_0(purged)
    renumbered = renumber_params(dropped, delta=-1)
    rebuilt = {**renumbered, 'map': rebuild_map(renumbered['map'], dropped_idx)}
    return rebuilt


def remove_rdkit_atom_0(m: Chem.Mol) -> Chem.Mol:
    return remove_rdkit_atom(m, 0)


def remove_rdkit_atom(m: Chem.Mol, atom_index: int) -> Chem.Mol:
    rw = Chem.RWMol(m)
    rw.RemoveAtom(atom_index)
    m2 = rw.GetMol()
    Chem.SanitizeMol(m2, catchErrors=True)
    return m2


def zmat_to_coords(zmat: dict,
                   keep_dummy: bool = False,
                   skip_undefined: bool = False,
                   ) -> Tuple[List[dict], List[str]]:
    """
    Generate the cartesian coordinates from a zmat dict.
    Considers the zmat atomic map so the returned coordinates is ordered correctly.
    Most common isotopes assumed, if this is not the case, then isotopes should be reassigned to the xyz.
    This function assumes that all zmat variables relate to already defined atoms with a lower index in the zmat.

    This function implements the SN-NeRF algorithm as described in:
    J. Parsons, J.B. Holmes, J.M Rojas, J. Tsai, C.E.M. Strauss, "Practical Conversion from Torsion Space to Cartesian
    Space for In Silico Protein Synthesis", Journal of Computational Chemistry 2005, 26 (10), 1063-1068,
    https://doi.org/10.1002/jcc.20237

    Tested in converterTest.py rather than zmatTest

    Args:
        zmat (dict): The zmat.
        keep_dummy (bool): Whether to keep dummy atoms ('X'), ``True`` to keep, default is ``False``.
        skip_undefined (bool): Whether to skip atoms with undefined variables, instead of raising an error.
                               ``True`` to skip, default is ``False``.

    Raises:
        ZMatError: If zmat is of wrong type or does not contain all keys.

    Returns: Tuple[List[dict], List[str]]
        - The cartesian coordinates.
        - The atomic symbols corresponding to the coordinates.
    """
    if not isinstance(zmat, dict):
        raise ZMatError(f'zmat has to be a dictionary, got {type(zmat)}')
    if 'symbols' not in zmat or 'coords' not in zmat or 'vars' not in zmat or 'map' not in zmat:
        raise ZMatError(f'Expected to find symbols, coords, vars, and map in zmat, got instead: {list(zmat.keys())}.')
    if not len(zmat['symbols']) == len(zmat['coords']) == len(zmat['map']):
        raise ZMatError(f'zmat sections symbols, coords, and map have different lengths: {len(zmat["symbols"])}, '
                        f'{len(zmat["coords"])}, and {len(zmat["map"])}, respectively.')
    for key, value in zmat['vars'].items():
        if value is None:
            raise ZMatError(f'Got ``None`` for var {key} in zmat:\n{zmat}')
    var_list = list(zmat['vars'].keys())
    coords_to_skip = list()
    for i, coords in enumerate(zmat['coords']):
        for coord in coords:
            if coord is not None and coord not in var_list:
                if skip_undefined:
                    coords_to_skip.append(i)
                else:
                    raise ZMatError(f'The parameter {coord} was not found in the "vars" section of '
                                    f'the zmat:\n{zmat["vars"]}')

    coords = list()
    for i in range(len(zmat['symbols'])):
        coords = _add_nth_atom_to_coords(zmat=zmat, coords=coords, i=i, coords_to_skip=coords_to_skip)

    # Reorder the xyz according to the zmat map and remove dummy atoms if requested.
    ordered_coords, ordered_symbols = list(), list()
    for i in range(len(zmat['symbols'])):
        zmat_index = key_by_val(zmat['map'], i)
        if zmat_index < len(coords) and i not in coords_to_skip and (zmat['symbols'][zmat_index] != 'X' or keep_dummy):
            ordered_coords.append(coords[zmat_index])
            ordered_symbols.append(zmat['symbols'][zmat_index])

    return ordered_coords, ordered_symbols

def purge_references_to_atom_0(zmat: dict) -> dict:
    """
    Replace any Z-matrix parameter referencing atom 0 with valid alternatives.
    Ensures only atoms with index < current are used as references (Z-matrix rule).
    Leaves map untouched. Atom 0 is still present and must be removed later.
    """
    z0 = zmat.copy()
    xyz_coords, _ = zmat_to_coords(zmat=z0, keep_dummy=True)
    new_vars, new_coords = dict(), list()

    def safe_calc_param(atoms):
        try:
            return calculate_param(coords=xyz_coords, atoms=atoms)
        except (ValueError, IndexError, VectorsError):
            return None

    all_param_names_used = set()

    for i, row in enumerate(z0['coords']):
        new_row = list()
        for p in row:
            if not isinstance(p, str):
                new_row.append(p)
                continue

            all_param_names_used.add(p)

            if '_0' not in p:
                new_row.append(p)
                continue

            groups = get_atom_indices_from_zmat_parameter(p)
            flat = [idx for group in groups for idx in group]

            if 0 not in flat:
                new_row.append(p)
                continue

            # Replace 0 with a valid reference
            used = set(flat) - {0}
            candidate = 1
            while candidate in used or candidate >= i:
                candidate += 1
                if candidate >= i:
                    p = None
                    break
            else:
                updated = [candidate if x == 0 else x for x in flat]
                updated[0] = i  # ensure proper first index
                tag = p.split("_")[0]
                new_p = "_".join([tag] + [str(idx) for idx in updated])

                try:
                    xyz_indices = [zmat['map'][j] for j in updated]
                except KeyError:
                    param_value = None
                else:
                    param_value = safe_calc_param(xyz_indices)

                if param_value is not None:
                    new_vars[new_p] = param_value
                    p = new_p
                else:
                    p = None

            new_row.append(p)

        new_coords.append(tuple(new_row))

    # Add original vars that are still used
    for k, v in z0['vars'].items():
        if k in all_param_names_used and k not in new_vars:
            new_vars[k] = v

    return {'symbols': z0['symbols'],
            'coords': tuple(new_coords),
            'vars': new_vars,
            'map': z0['map']}



def relocate_zmat_dummy_atoms_to_the_end(zmat_map: dict) -> dict:
    """
    Relocate all dummy atoms in a ZMat to the end of the corresponding Cartesian coordinates atom list.
    Only modifies the values of the ZMat map.

    Args:
        zmat_map (dict): The ZMat map.

    Returns:
        dict: The updated ZMat map.
    """
    no_x_map = {key: val for key, val in zmat_map.items() if isinstance(val, int)}
    x_map = {key: val for key, val in zmat_map.items() if isinstance(val, str) and 'X' in val}
    for x_atom_number_in_zmat, x_atom_val in x_map.items():
        x_atom_number_in_cartesian = int(x_atom_val[1:])
        if any(x_atom_number_in_cartesian < number_in_cartesian for number_in_cartesian in no_x_map.values()):
            no_x_map = {key: val - 1 if x_atom_number_in_cartesian < val else val for key, val in no_x_map.items()}
    num_no_x_atoms = len(no_x_map.keys())
    x_map = {key: f'X{num_no_x_atoms + i}' for i, (key, val) in enumerate(x_map.items())}
    no_x_map.update(x_map)
    return no_x_map

def update_new_map_based_on_zmat_2(new_map: dict,
                                   zmat_2: dict,
                                   num_atoms_1,
                                   atom_map: dict,
                                   reactants_reversed: bool = False,
                                   ):
    """
    Update the map for the combined zmats, here only consider the modified version of ``zmat_2``.
    This function assumes that all dummy atoms are located at the end of the respective Cartesian coordinates
    for ``zmat_2`` (i.e., that relocate_zmat_dummy_atoms_to_the_end() was called).

    Args:
        new_map (dict): The initial map for the combined zmats based on ``zmat_1``.
        zmat_2 (dict): The modified ``zmat_2`` (with the 1st atom removed and all dummy atoms relocated to the end).
        num_atoms_1 (int): The number of atoms in ``zmat_1``.
        atom_map (dict): The atom-map relating the product that corresponds with ``zmat_2`` to the reactant molecule.
        reactants_reversed (bool, optional): Whether the reactants were reversed relative to the RMG template.

    Returns:
        dict: The updated map for the combined zmats.
    """
    if atom_map is None:
        raise ValueError('Could not generate a combined zmat map without an atom_map.')
    key_inc = num_atoms_1
    val_inc = 0 if reactants_reversed else num_atoms_1
    dummy_atom_counter = 0
    num_of_non_x_atoms_in_zmat_2 = len([val for val in zmat_2['map'].values() if isinstance(val, int)])
    for key, val in zmat_2['map'].items():
        # Atoms in zmat_2 always come after atoms in zmat_1 in the new zmat, regardless of the reactants/products
        # order on each side of the given reaction.
        new_key = key + key_inc
        # Use the atom_map to map atoms in zmat_2 (i.e., values in zmat_2's 'map') to atoms in the R(*3)
        # **reactant** (at least for H-Abstraction reactions), since zmat_2 was built based on atoms in the R(*3)-H(*2)
        # **product** (at least for H-Abstraction reactions).
        if isinstance(val, str) and 'X' in val:
            # A dummy atom is not in the atom_map.
            new_val = num_of_non_x_atoms_in_zmat_2 + val_inc + dummy_atom_counter
            new_val = f'X{new_val}'
            dummy_atom_counter += 1
        else:
            new_val = atom_map[val] + val_inc
        new_map[new_key] = new_val
    return new_map




def drop_symbol_and_coords_row_0(zmat: dict) -> dict:
    """
    Remove the 0th atom from the Z-matrix:
    - Removes the first symbol and coordinate row
    - Clears DOFs for the new atoms 0–2 (Z-matrix rules)
    - Removes unused variables from `vars`
    - Leaves map untouched (to be rebuilt later)
    - Does NOT decrement atom indices; that’s handled later
    """
    z0 = zmat.copy()

    # Drop the first atom (index 0)
    new_symbols = tuple(z0['symbols'][1:])
    new_coords = list(z0['coords'][1:])

    # Reset DOFs for atoms 0–2
    if len(new_coords) >= 1:
        new_coords[0] = (None, None, None)
    if len(new_coords) >= 2:
        new_coords[1] = (new_coords[1][0], None, None)
    if len(new_coords) >= 3:
        new_coords[2] = (new_coords[2][0], new_coords[2][1], None)

    # Drop any vars no longer referenced
    used_keys = {p for row in new_coords for p in row if isinstance(p, str)}
    new_vars = {k: v for k, v in z0['vars'].items() if k in used_keys}

    return {'symbols': new_symbols,
            'coords': tuple(new_coords),
            'vars': new_vars,
            'map': z0['map'].copy()}  # leave untouched for rebuild_map()


def renumber_params(zmat: dict, delta: int = -1) -> dict:
    """
    Renumber all atom indices in param names in both `coords` and `vars`.
    Returns a self-consistent Z-matrix.

    Args:
        zmat (dict): The Z-matrix dict with 'coords' and 'vars'
        delta (int): The value to shift all indices by (usually -1)

    Returns:
        dict: The updated Z-matrix with renamed param strings and values.
    """
    pattern = re.compile(r'^([RAD]X?)_(\d+)(?:_(\d+))?(?:_(\d+))?(?:_(\d+))?')

    # Param key mapping: old_key → new_key
    key_map = {}

    # 1. Rename all parameter strings in coords
    new_coords = []
    for row in zmat['coords']:
        new_row = []
        for p in row:
            if not isinstance(p, str):
                new_row.append(p)
                continue
            match = pattern.fullmatch(p)
            if not match:
                new_row.append(p)
                continue
            tag, *idxs = match.groups()
            shifted = [str(max(int(i) + delta, 0)) for i in idxs if i is not None]
            new_key = '_'.join([tag] + shifted)
            key_map[p] = new_key
            new_row.append(new_key)
        new_coords.append(tuple(new_row))

    # 2. Rename all keys in vars that were used in coords
    new_vars = {}
    for old_key, val in zmat['vars'].items():
        new_key = key_map.get(old_key)
        if new_key:
            new_vars[new_key] = val

    return {**zmat,
            'coords': tuple(new_coords),
            'vars': new_vars}


def up_param(param: str,
             increment: Optional[int] = None,
             increment_list: Optional[List[int]] = None,
             ) -> str:
    """
    Increase the indices represented by a zmat parameter.

    Args:
        param (str): The zmat parameter, e.g., 'D_4_2_1_0'.
        increment (int, optional): Uniform value to add to each index.
        increment_list (list, optional): Individual increments per index.

    Raises:
        ZMatError: If no increment was provided or resulting index is negative.

    Returns: str
        The new parameter with increased indices.
    """
    if increment is None and increment_list is None:
        raise ZMatError('Either increment or increment_list must be specified.')
    indices = get_atom_indices_from_zmat_parameter(param)[0]
    if increment_list:
        if len(increment_list) != len(indices):
            raise ZMatError(f'Increment list length ({len(increment_list)}) does not match index count ({len(indices)})')
        new_indices = [i + inc for i, inc in zip(indices, increment_list)]
    else:
        new_indices = [i + increment for i in indices]
    if any(i < 0 for i in new_indices):
        raise ZMatError(f'Negative index in param "{param}" after increment.')
    tag = param.split('_')[0]
    return '_'.join([tag] + [str(i) for i in new_indices])

def _choose_best_by_rmsd(m1: Chem.Mol, m2: Chem.Mol, matches: Iterable[Tuple[int, ...]]) -> Tuple[int, ...]:
    """
    If multiple isomorphisms exist, choose the one that best aligns geometry (lowest RMSD)
    assuming both mols have conformer 0 in corresponding coordinate frames (or close enough).
    """
    best = None
    best_rmsd = float("inf")
    for match in matches:
        # match is a tuple: mol2_index_for_mol1_atom0, mol2_index_for_mol1_atom1, ...
        atomMap = [(i, int(match[i])) for i in range(len(match))]
        try:
            rmsd = rdMolAlign.AlignMol(m2, m1, atomMap=atomMap)  # aligns m2 onto m1
        except Exception:
            continue
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best = match
    if best is None:
        # fallback: just take first
        return next(iter(matches))
    return best


def key_by_val(dictionary: dict,
               value: Any,
               ) -> Any:
    """
    A helper function for getting a key from a dictionary corresponding to a certain value.
    Does not check for value unicity.

    Args:
        dictionary (dict): The dictionary.
        value: The value.

    Raises:
        ValueError: If the value could not be found in the dictionary.

    Returns: Any
        The key.
    """
    for key, val in dictionary.items():
        if val == value or (isinstance(value, int) and val == f'X{value}'):
            return key
    raise ValueError(f'Could not find value {value} in the dictionary\n{dictionary}')


def check_atom_r_constraints(atom_index: int,
                             constraints: dict
                             ) -> Tuple[Optional[tuple], Optional[str]]:
    """
    Check distance constraints for an atom. 
    'R' constraints are a list of tuples with length 2.
    The first atom, in an R constraint, is considered "constraints" for zmat generation
    Its distance will be relative to the second atom.
    
    :param atom_index: Description
    :type atom_index: int
    :param constraints: Description
    :type constraints: dict
    :return: Description
    :rtype: Tuple[tuple | None, str | None]
    """
    if not any(k.startswith("R_") for k in constraints.keys()):
        return None, None

    r_constraints = {k: v for k, v in constraints.items() if k.startswith("R_")}

    for r_constraint_list in r_constraints.values():
        if any(len(tpl) != 2 for tpl in r_constraint_list):
            raise ZMatError(f'"R" constraints must contain only tuples of length two, got: {r_constraints}.')

    if any(k not in ("R_atom", "R_group") for k in r_constraints.keys()):
        raise ZMatError(f'"R" constraints must be either "R_atom" or "R_group", got: {r_constraints}.')

    occurrences = 0
    for r_constraint_list in r_constraints.values():
        occurrences += sum(tpl[0] == atom_index for tpl in r_constraint_list)

    if occurrences == 0:
        return None, None
    if occurrences > 1:
        raise ZMatError(
            f'A single atom cannot be constrained more than once. '
            f'Atom {atom_index} is constrained {occurrences} times in "R" constraints.'
        )

    for constraint_type, r_constraint_list in r_constraints.items():
        for tpl in r_constraint_list:
            if tpl[0] == atom_index:
                return (tpl[0], tpl[1]), constraint_type

    return None, None


def check_atom_a_constraints(atom_index: int,
                             constraints: dict) -> Tuple[Optional[tuple], Optional[str]]:
    """
    Check angle constraints for an atom
    'A' constraints are a list of tuples with length 3.
    The first atom in an A constraint is considered 'constraint' for zmat generation.
    Its angle will be relative to the second and thirm atom
    
    :param atom_index: Description
    :type atom_index: int
    :param constraints: Description
    :type constraints: dict
    :return: Description
    :rtype: Tuple[tuple | None, str | None]
    """
    if not any(k.startswith("A_") for k in constraints.keys()):
        return None, None

    a_constraints = {k: v for k, v in constraints.items() if k.startswith("A_")}

    for a_list in a_constraints.values():
        if any(len(tpl) != 3 for tpl in a_list):
            raise ZMatError(f'"A" constraints must contain only tuples of length three, got: {a_constraints}.')

    if any(k not in ("A_atom", "A_group") for k in a_constraints.keys()):
        raise ZMatError(f'"A" constraints must be either "A_atom" or "A_group", got: {a_constraints}.')

    occurrences = 0
    for a_list in a_constraints.values():
        occurrences += sum(tpl[0] == atom_index for tpl in a_list)

    if occurrences == 0:
        return None, None
    if occurrences > 1:
        raise ZMatError(
            f'A single atom cannot be constrained more than once. '
            f'Atom {atom_index} is constrained {occurrences} times in "A" constraints.'
        )

    for constraint_type, a_list in a_constraints.items():
        for tpl in a_list:
            if tpl[0] == atom_index:
                return (tpl[0], tpl[1], tpl[2]), constraint_type

    return None, None



def check_atom_d_constraints(atom_index: int,
                             constraints: Dict[str, List[Tuple[int, ...]]],
                             ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[str]]:
    if not any(k.startswith("D_") for k in constraints.keys()):
        return None, None

    d_constraints = {k: v for k, v in constraints.items() if k.startswith("D_")}

    for d_list in d_constraints.values():
        if any(len(tpl) != 4 for tpl in d_list):
            raise ZMatError(f'"D" constraints must contain only tuples of length four, got: {d_constraints}.')

    if any(k not in ("D_atom", "D_group") for k in d_constraints.keys()):
        raise ZMatError(f'"D" constraints must be either "D_atom" or "D_group", got: {d_constraints}.')

    occurrences = 0
    for d_list in d_constraints.values():
        occurrences += sum(tpl[0] == atom_index for tpl in d_list)

    if occurrences == 0:
        return None, None
    if occurrences > 1:
        raise ZMatError(
            f'A single atom cannot be constrained more than once. '
            f'Atom {atom_index} is constrained {occurrences} times in "D" constraints.'
        )

    for constraint_type, d_list in d_constraints.items():
        for tpl in d_list:
            if tpl[0] == atom_index:
                return (tpl[0], tpl[1], tpl[2], tpl[3]), constraint_type

    return None, None


def is_atom_in_new_fragment(
    atom_index: int,
    zmat: Dict[str, Union[dict, tuple]],
    fragments: Optional[List[List[int]]] = None,
    skip_atoms: Optional[List[int]] = None ) -> bool:
    """
    
    """
    skip_atoms = skip_atoms or []
    if not fragments or len(fragments) <= 1:
        return False
    
    mapped_atoms: Set[int] = {
        frag_idx for z_idx, frag_idx in zmat["map"].items()
    }
    
    for fragment in fragments:
        if atom_index in fragment:
            # if non of the already mapped atoms are in this fragment, it's a new fragment
            if mapped_atoms.isdisjoint(fragment):
                return True
    return False
    


def determine_r_atoms(
    zmat: Dict[str, Union[dict, tuple]],
    xyz: Dict[str, tuple],
    connectivity: Optional[Dict[int, Iterable[int]]],
    n: int,
    atom_index: int,
    r_constraint: Optional[Tuple[int, int]] = None,
    a_constraint: Optional[Tuple[int, int, int]] = None,
    d_constraint: Optional[Tuple[int, int, int, int]] = None,
    trivial_assignment: bool = False,
    fragments: Optional[List[List[int]]] = None,
) -> Optional[List[int]]:
    """
    Determine the atoms for defining the distance R.
    Return either:
      - None (for the very first atom), or
      - [n, ref] where ref is a z-matrix index already present in zmat['map'].

    Notes:
      * n is the zmat index being added.
      * atom_index is the xyz/mol index of that same atom (mapped later via zmat['map']).
      * connectivity maps xyz index -> list of neighboring xyz indices (e.g., from RDKit bonds).
    """
    # If we’re starting a new fragment (TS / vdW well), connectivity across fragments is meaningless.
    if is_atom_in_new_fragment(atom_index=atom_index, zmat=zmat, fragments=fragments):
        connectivity = None

    # 0) First atom has no R definition
    if len(zmat["coords"]) == 0:
        r_atoms: Optional[List[int]] = None

    # 1) If constrained (R/A/D), always use the constraint’s anchor (second atom in the tuple)
    elif any(constraint is not None for constraint in (r_constraint, a_constraint, d_constraint)):
        if r_constraint is not None:
            r_atoms = [n, key_by_val(zmat["map"], r_constraint[1])]
        elif a_constraint is not None:
            r_atoms = [n, key_by_val(zmat["map"], a_constraint[1])]
        else:  # d_constraint is not None
            r_atoms = [n, key_by_val(zmat["map"], d_constraint[1])]

    # 2) Otherwise, prefer a connectivity-based reference (pick a good already-added neighbor side)
    elif connectivity is not None:
        r_atoms = [n]

        # Keys: neighbor xyz index; values: (depth, linear)
        # depth = how many atoms we can “see” going away from atom_index via that neighbor
        # linear = whether that explored chain appears linear (bad for defining angles/dihedrals)
        atom_dict: Dict[int, Tuple[int, bool]] = {}

        for atom_c in connectivity.get(atom_index, []):
            # Adopted from arc.common.determine_top_group_indices()
            linear = True
            explored_atom_list = [atom_index]
            atom_list_to_explore1 = [atom_c]
            atom_list_to_explore2: List[int] = []
            top: List[int] = []

            while len(atom_list_to_explore1) + len(atom_list_to_explore2):
                for atom3 in atom_list_to_explore1:
                    top.append(atom3)

                    # Expand from atom3
                    if atom3 in connectivity:
                        for atom4 in connectivity[atom3]:
                            if atom4 not in explored_atom_list and atom4 not in atom_list_to_explore2:
                                if xyz["symbols"][atom4] in ["H", "F", "Cl", "Br", "I", "X"]:
                                    # terminal-ish: append but don’t expand further
                                    top.append(atom4)
                                else:
                                    atom_list_to_explore2.append(atom4)

                    explored_atom_list.append(atom3)

                atom_list_to_explore1, atom_list_to_explore2 = atom_list_to_explore2, []

                # Once we have at least 2 atoms in 'top', we can evaluate linearity
                if len(top) >= 2:
                    angle = calculate_param(
                        coords=xyz["coords"],
                        atoms=[atom_index] + top[-2:],  # [center] + last two in the explored direction
                        index=0,
                    )
                    if not is_angle_linear(angle, tolerance=TOL_180):
                        linear = False
                        if len(top) >= 3:
                            # Non-linear and long enough -> good information; stop early.
                            break

            atom_dict[atom_c] = (len(top), linear)

        # Rank candidate references (must already be in zmat, and not dummy)
        long_non_linear: List[int] = []
        long_linear: List[int] = []
        two_non_linear: List[int] = []
        two_linear: List[int] = []
        one: List[int] = []

        for atom_c, (depth, is_lin) in atom_dict.items():
            if atom_c in list(zmat["map"].values()):
                zmat_index_c = key_by_val(zmat["map"], atom_c)
                if not is_dummy(zmat, zmat_index_c):
                    if depth >= 3 and not is_lin:
                        long_non_linear.append(zmat_index_c)
                    elif depth >= 3 and is_lin:
                        long_linear.append(zmat_index_c)
                    elif depth == 2 and not is_lin:
                        two_non_linear.append(zmat_index_c)
                    elif depth == 2 and is_lin:
                        two_linear.append(zmat_index_c)
                    elif depth == 1:
                        one.append(zmat_index_c)

        if long_non_linear:
            r_atoms.append(long_non_linear[0])
        elif two_non_linear:
            r_atoms.append(two_non_linear[0])
        elif long_linear:
            r_atoms.append(long_linear[0])
        elif two_linear:
            r_atoms.append(two_linear[0])
        elif one:
            r_atoms.append(one[0])

        # If we didn’t end up with two unique atoms, fall back to trivial assignment
        if len(set(r_atoms)) != 2:
            trivial_assignment = True

    # 3) No constraints and no connectivity -> trivial assignment
    else:
        trivial_assignment = True
        r_atoms = []

    if trivial_assignment and isinstance(r_atoms, list) and len(r_atoms) != 2:
        r_atoms = [n]

        if len(zmat["coords"]) in (1, 2):
            # for 2nd/3rd atom, just reference the last added
            r_atoms.append(len(zmat["coords"]) - 1)
        else:
            # otherwise reference the most recent non-dummy atom
            for i in reversed(range(n)):
                if not is_dummy(zmat, i):
                    r_atoms.append(i)
                    break

        if len(set(r_atoms)) != 2:
            raise ZMatError(f"Could not come up with two unique non-dummy r_atoms (r_atoms = {r_atoms}).")

    # Sanity checks: reference must already be in zmat and must be unique
    if r_atoms is not None and r_atoms[-1] not in list(zmat["map"].keys()):
        raise ZMatError(
            f"The reference R atom {r_atoms[-1]} for the index atom {atom_index} has not been "
            f"added to the zmat yet. Added atoms are (zmat index: xyz index): {zmat['map']}."
        )
    if r_atoms is not None and len(set(r_atoms)) != 2:
        raise ZMatError(f"Could not come up with two unique r_atoms (r_atoms = {r_atoms}).")

    return r_atoms


def is_dummy(zmat: dict,
             zmat_index: int,
             ) -> bool:
    """
    Determine whether an atom in a zmat is a dummy atom by its zmat index.

    Args:
        zmat (dict): The zmat with symbol and map information.
        zmat_index (int): The atom index in the zmat.

    Raises:
        ZMatError: If the index is invalid.

    Returns:
        bool: Whether the atom represents a dummy atom 'X'. ``True`` if it does.
    """
    if len(zmat['symbols']) <= zmat_index:
        raise ZMatError(f'index {zmat_index} is invalid for a zmat with only {len(zmat["symbols"])} atoms')
    return zmat['symbols'][zmat_index] == 'X'



def is_angle_linear(angle: float,
                    tolerance: float = 0.9,
                    ) -> bool:
    """
    Check whether an angle is close to 180 or 0 degrees.

    Args:
        angle (float): The angle in degrees.
        tolerance (float): The tolerance to consider.

    Returns:
        bool: Whether the angle is close to 180 or 0 degrees, ``True`` if it is.
    """
    return (180 - tolerance < angle <= 180) or (0 <= angle < tolerance)


def _pick_non_linear_fourth(
    zmat: Dict[str, Union[dict, tuple]],
    coords: Union[list, tuple],
    backbone_z: List[int],   # [j_z, k_z] in your convention (d_atoms will be [n, j, k, A])
    n: int,
    allow_dummies: bool,
) -> Optional[int]:
    """Scan backwards to find a zmat index A that makes angle(A - k - j) non-linear."""
    j_z, k_z = backbone_z[0], backbone_z[1]

    for a_z in reversed(range(n)):
        if a_z in (n, j_z, k_z):
            continue
        if a_z not in zmat["map"]:
            continue
        if (not allow_dummies) and (a_z < len(zmat["symbols"]) and is_dummy(zmat, a_z)):
            continue

        try:
            a_xyz = zmat["map"][a_z]
            j_xyz = zmat["map"][j_z]
            k_xyz = zmat["map"][k_z]
            if isinstance(a_xyz, str) or isinstance(j_xyz, str) or isinstance(k_xyz, str):
                continue
            ang = calculate_param(coords=coords, atoms=[a_xyz, k_xyz, j_xyz])
        except VectorsError:
            continue

        if not is_angle_linear(ang, tolerance=TOL_180):
            return a_z

    return None



def determine_d_atoms(
    zmat: Dict[str, Union[dict, tuple]],
    xyz: Dict[str, tuple],
    coords: Union[list, tuple],
    connectivity: Optional[Dict[int, Iterable[int]]],
    a_atoms: Optional[List[int]],
    n: int,
    atom_index: int,
    d_constraint: Optional[Tuple[int, int, int, int]] = None,
    d_constraint_type: Optional[str] = None,
    specific_atom: Optional[int] = None,
    dummy: bool = False,
    fragments: Optional[List[List[int]]] = None,
) -> Optional[List[int]]:
    if a_atoms is not None and is_atom_in_new_fragment(
        atom_index=atom_index, zmat=zmat, fragments=fragments, skip_atoms=a_atoms
    ):
        connectivity = None
    if a_atoms is not None and len(a_atoms) != 3:
        raise ZMatError(f"a_atoms must be a list of length 3, got {a_atoms}")
    if len(zmat["coords"]) <= 2:
        d_atoms = None
    elif d_constraint is not None:
        if d_constraint_type not in ["D_atom", "D_group"]:
            raise ZMatError(f'Got an invalid D constraint type "{d_constraint_type}" for {d_constraint}')
        d_atoms = [n] + [key_by_val(zmat["map"], atom) for atom in d_constraint[1:]]
    elif specific_atom is not None and a_atoms is not None:
        if not isinstance(specific_atom, int):
            raise ZMatError(f"specific atom must be of type int, got {type(specific_atom)}")
        d_atoms = a_atoms + [specific_atom]
    elif connectivity is not None:
        d_atoms = determine_d_atoms_from_connectivity(
            zmat, xyz, coords, connectivity, a_atoms, atom_index, dummy=dummy, allow_a_to_be_dummy=False
        )
        if len(d_atoms) < 4:
            d_atoms = determine_d_atoms_without_connectivity(zmat, coords, a_atoms, n)
            if len(d_atoms) < 4:
                d_atoms = determine_d_atoms_from_connectivity(
                    zmat, xyz, coords, connectivity, a_atoms, atom_index, dummy=dummy, allow_a_to_be_dummy=True
                )
    else:
        d_atoms = determine_d_atoms_without_connectivity(zmat, coords, a_atoms, n)
    if d_atoms is not None:
        if len(d_atoms) < 4:
            for i in reversed(range(len(xyz["symbols"]))):
                if i not in d_atoms and i in list(zmat["map"].keys()):
                    angle = calculate_param(coords=coords, atoms=[zmat["map"][z_index] for z_index in d_atoms[1:] + [i]])
                    if not is_angle_linear(angle, tolerance=TOL_180):
                        d_atoms.append(i)
                        break
        if len(set(d_atoms)) != 4:
            logger.error(f"Could not come up with four unique d_atoms (d_atoms = {d_atoms}). "
                         f"Setting d_atoms to [{n}, 2, 1, 0]")
            d_atoms = [n, 2, 1, 0]
        if any(d_atom not in list(zmat["map"].keys()) for d_atom in d_atoms[1:]):
            raise ZMatError(
                f"A reference D atom in {d_atoms} for the index atom {atom_index} has not been "
                f"added to the zmat yet. Added atoms are (zmat index: xyz index): {zmat['map']}."
            )
    return d_atoms


def determine_d_atoms_without_connectivity(
    zmat: dict,
    coords: Union[list, tuple],
    a_atoms: list,
    n: int,
) -> list:
    d_atoms = [atom for atom in a_atoms]
    for i in reversed(range(n)):
        if i not in d_atoms and i in list(zmat["map"].keys()) and (i >= len(zmat["symbols"]) or not is_dummy(zmat, i)):
            try:
                angle = calculate_param(coords=coords, atoms=[zmat["map"][z_index] for z_index in d_atoms[1:] + [i]])
            except VectorsError:
                continue
            if not is_angle_linear(angle, tolerance=TOL_180):
                d_atoms.append(i)
                break
    if len(d_atoms) < 4:
        for i in reversed(range(n)):
            if i not in d_atoms and i in list(zmat["map"].keys()):
                try:
                    angle = calculate_param(coords=coords, atoms=[zmat["map"][z_index] for z_index in d_atoms[1:] + [i]])
                except VectorsError:
                    continue
                if not is_angle_linear(angle, tolerance=TOL_180):
                    d_atoms.append(i)
                    break
    return d_atoms


def determine_d_atoms_from_connectivity(
    zmat: dict,
    xyz: dict,
    coords: Union[list, tuple],
    connectivity: Dict[int, Iterable[int]],
    a_atoms: list,
    atom_index: int,
    dummy: bool = False,
    allow_a_to_be_dummy: bool = False,
) -> list:
    d_atoms = [atom for atom in a_atoms]
    int_refs = [i for i in d_atoms if isinstance(i, int)]
    if len(int_refs) < 2:
        return d_atoms
    ref_last, ref_penult = int_refs[-1], int_refs[-2]
    for atom in (
        list(connectivity.get(zmat["map"][ref_last], []))
        + list(connectivity.get(zmat["map"][ref_penult], []))
        + list(connectivity.get(atom_index, []))
    ):
        if atom != atom_index and atom in list(zmat["map"].values()) and (
            not is_dummy(zmat, key_by_val(zmat["map"], atom)) or (not dummy and allow_a_to_be_dummy)
        ):
            zmat_index = None
            if atom not in list([zmat["map"][d_atom] for d_atom in d_atoms[1:]]):
                i = 0
                atom_a, atom_b, atom_c = atom, zmat["map"][d_atoms[2]], zmat["map"][d_atoms[1]]
                while i < len(list(connectivity.keys())):
                    angle = calculate_param(coords=coords, atoms=[atom_a, atom_b, atom_c])
                    if is_angle_linear(angle, tolerance=TOL_180):
                        num_of_neighbors = len(list(connectivity.get(atom_a, [])))
                        if num_of_neighbors == 1:
                            b_neighbors = list(connectivity.get(atom_b, []))
                            x_neighbor = [neighbor for neighbor in b_neighbors if xyz["symbols"][neighbor] == "X"][0]
                            if key_by_val(zmat["map"], f"X{x_neighbor}") not in d_atoms:
                                zmat_index = key_by_val(zmat["map"], f"X{x_neighbor}")
                                break
                        elif num_of_neighbors == 2:
                            a_neighbors = list(connectivity.get(atom_a, []))
                            atom_e = a_neighbors[0] if a_neighbors[0] != atom_b else a_neighbors[1]
                            if atom_e in list(zmat["map"].values()):
                                angle = calculate_param(coords=coords, atoms=[atom_e, atom_b, atom_c])
                                if is_angle_linear(angle, tolerance=TOL_180):
                                    atom_a = atom_e
                                elif key_by_val(zmat["map"], atom_e) not in d_atoms:
                                    zmat_index = key_by_val(zmat["map"], atom_e)
                                    break
                        elif num_of_neighbors > 2:
                            for a_neighbor in connectivity.get(atom_a, []):
                                if a_neighbor != atom_b:
                                    angle = calculate_param(coords=coords, atoms=[a_neighbor, atom_b, atom_c])
                                    if (
                                        not is_angle_linear(angle, tolerance=TOL_180)
                                        and a_neighbor in list(zmat["map"].values())
                                        and key_by_val(zmat["map"], a_neighbor) not in d_atoms
                                    ):
                                        zmat_index = key_by_val(zmat["map"], a_neighbor)
                                        break
                    elif atom_a in list(zmat["map"].values()):
                        zmat_index = key_by_val(zmat["map"], atom_a)
                        break
                    i += 1
            if zmat_index is None and len(d_atoms) == 3 and "X" in zmat["symbols"] and not dummy and allow_a_to_be_dummy:
                dummies = [(key, int(val[1:])) for key, val in zmat["map"].items() if re.match(r"X\\d", str(val))]
                for dummy_entry in dummies:
                    zmat_index = dummy_entry[0]
            if zmat_index is not None:
                d_atoms.append(zmat_index)
            break
    if len(d_atoms) == 3 and len(list(connectivity.get(atom_index, []))) > 2:
        third = list(connectivity.get(atom_index, []))[2]
        if third in list(zmat["map"].values()) and third not in [zmat["map"][d_atom] for d_atom in d_atoms[1:]]:
            angle = calculate_param(
                coords=coords, atoms=[zmat["map"][d_atom] for d_atom in d_atoms[1:]] + [third]
            )
            if (
                not is_angle_linear(angle, tolerance=TOL_180)
                and third in list(zmat["map"].values())
                and key_by_val(zmat["map"], third) not in d_atoms
            ):
                d_atoms.append(key_by_val(zmat["map"], third))
    return d_atoms



def _has_dummy_ref(zmat: dict, ref_atoms: list) -> bool:
    """
    Return True if any reference atom in `ref_atoms` is a dummy atom (symbol == 'X').

    Notes:
        - `ref_atoms` are z-matrix indices (not xyz indices).
        - This assumes dummy atoms are represented by symbol 'X' in zmat['symbols'].
    """
    return any(zmat["symbols"][z_idx] == "X" for z_idx in ref_atoms)


def update_zmat_with_new_atom(zmat: dict,
                              xyz: dict,
                              coords: Union[list, tuple],
                              n: int,
                              atom_index: int,
                              r_atoms: list,
                              a_atoms: list,
                              d_atoms: list,
                              added_dummy: bool = False,
                              ) -> dict:
    """
    Update the zmat with a new atom.
    (Same logic as your original, but with clearer dummy detection.)
    """
    zmat["symbols"].append(xyz["symbols"][atom_index])

    if atom_index in zmat["map"].values():
        raise ZMatError(
            f"Cannot assign atom {atom_index} to key {n}, it is already in the zmat map: {zmat['map']}"
        )

    zmat["map"][n] = atom_index

    # --- build variable strings ---
    if r_atoms is None:
        r_str = None
    else:
        x = "X" if _has_dummy_ref(zmat, r_atoms) else ""
        r_str = f"R{x}_{r_atoms[0]}_{r_atoms[1]}"

    if a_atoms is None:
        a_str = None
    else:
        x = "X" if _has_dummy_ref(zmat, a_atoms) else ""
        a_str = f"A{x}_{a_atoms[0]}_{a_atoms[1]}_{a_atoms[2]}"

    if d_atoms is None:
        d_str = None
    else:
        x = "X" if _has_dummy_ref(zmat, d_atoms) else ""
        d_str = f"D{x}_{d_atoms[0]}_{d_atoms[1]}_{d_atoms[2]}_{d_atoms[3]}"

    # --- safety: no overwriting vars ---
    for string in (r_str, a_str, d_str):
        if string is not None and string in zmat["vars"]:
            raise ZMatError(f"{string} is already in vars: {zmat['vars']}")

    zmat["coords"].append((r_str, a_str, d_str))

    # --- sanity: no repeated indices within each atom spec list ---
    if any(
        atoms is not None and any(atoms.count(atom) > 1 for atom in atoms)
        for atoms in (r_atoms, a_atoms, d_atoms)
    ):
        raise ZMatError(
            "zmat atom specifications must not have repetitions, got:\n"
            f"r_atoms={r_atoms}, a_atoms={a_atoms}, d_atoms={d_atoms}"
        )

    # --- compute numeric values ---
    if r_atoms is not None:
        zmat["vars"][r_str] = calculate_param(coords=coords, atoms=[zmat["map"][z_i] for z_i in r_atoms])

    if added_dummy:
        # dummy-assisted placement convention
        zmat["vars"][a_str] = 90.0

        # The dihedral angle could be either 0 or 180 degrees, depends on the relative position of atom D and B, C
        # d_atoms represent the zmat indices of atoms D, C, X, and B (per your original comment/convention).
        bcd_angle = calculate_param(
            coords=coords,
            atoms=[zmat["map"][d_atoms[3]], zmat["map"][d_atoms[1]], zmat["map"][d_atoms[0]]],
            index=0,
        )

        if 180 - TOL_180 < bcd_angle <= 180:
            zmat["vars"][d_str] = 180.0
        elif 0 <= bcd_angle < TOL_180:
            zmat["vars"][d_str] = 0.0
        else:
            raise ZMatError(
                f"Atoms {d_atoms} for a non-linear sequence with an angle of {bcd_angle}. "
                f"Expected a linear sequence when using a dummy atom."
            )

    else:
        if a_atoms is not None:
            zmat["vars"][a_str] = calculate_param(coords=coords, atoms=[zmat["map"][z_i] for z_i in a_atoms])
        if d_atoms is not None:
            zmat["vars"][d_str] = calculate_param(coords=coords, atoms=[zmat["map"][z_i] for z_i in d_atoms])

    return zmat


def add_dummy_atom(zmat: dict,
                   xyz: dict,
                   coords: Union[list, tuple],
                   connectivity: dict,
                   r_atoms: list,
                   a_atoms: list,
                   n: int,
                   atom_index: int,
                   ) -> Tuple[dict, list, int, list, list, int]:
    """
    Add a dummy atom 'X' to the zmat.
    Also updates the r_atoms and a_atoms lists for the original (non-dummy) atom.

    Args:
        zmat (dict): The zmat.
        xyz (dict): The xyz dict.
        coords (Union[list, tuple]): Just the 'coords' part of the xyz dict.
        connectivity (dict): The atoms connectivity (keys are indices in the mol/xyz).
        r_atoms (list): The determined r_atoms.
        a_atoms (list): The determined a_atoms.
        n (int): The 0-index of the atom in the zmat to be added.
        atom_index (int): The 0-index of the atom in the molecule or cartesian coordinates to be added.
                          (``n`` and ``atom_index`` refer to the same atom, but it might have different indices
                          in the zmat and the molecule/xyz)

    Returns:
        Tuple[dict, list, int, list, list, int]:
            - The zmat.
            - The coordinates (list of tuples).
            - The updated atom index in the zmat.
            - The R atom indices.
            - The A atom indices.
            - A specific atom index to be used as the last entry of the D atom indices.
    """
    zmat['symbols'].append('X')
    zmat['map'][n] = f'X{len(xyz["symbols"])}'
    xyz['symbols'] += ('X',)
    xyz['isotopes'] += ('None',)

    # Determine the atoms for defining the dihedral angle, D, **for the dummy atom, X**.
    d_atoms = determine_d_atoms(zmat, xyz, coords, connectivity, a_atoms, n, atom_index, dummy=True)
    r_str = f'RX_{r_atoms[0]}_{r_atoms[1]}'
    a_str = f'AX_{a_atoms[0]}_{a_atoms[1]}_{a_atoms[2]}'
    d_str = f'DX_{d_atoms[0]}_{d_atoms[1]}_{d_atoms[2]}_{d_atoms[3]}' if d_atoms is not None else None
    zmat['coords'].append((r_str, a_str, d_str))  # the coords of the dummy atom
    zmat['vars'][r_str] = 1.0
    zmat['vars'][a_str] = 90.0
    if d_str is not None:
        zmat['vars'][d_str] = 180
    # Update xyz with the dummy atom (useful when this atom is used to define dihedrals of other atoms).
    coords = _add_nth_atom_to_coords(zmat=zmat, coords=list(coords), i=n)
    if connectivity is not None:
        # Update the connectivity dict to reflect that X is connected to the respective atom (r_atoms[1]),
        # this will help later in avoiding linear angles in the last three indices of a dihedral.
        connectivity[zmat['map'][r_atoms[1]]].append(int(zmat['map'][n][1:]))  # Take from 'X15'.
        connectivity[int(zmat['map'][n][1:])] = [zmat['map'][r_atoms[1]]]
    # Before adding the original (non-dummy) atom, increase n due to the increased number of atoms.
    n += 1
    # Store atom B's index for the dihedral of atom D.
    specific_last_d_atom = a_atoms[-1]
    # Update the r_atoms and a_atoms for the original (non-dummy) atom (d_atoms is set below).
    #
    #          X (dummy atom)
    #          |
    #     B -- C -- D (original atom)
    #   /
    #  A (optional)
    r_atoms = [n, r_atoms[1]]  # make this (D, C)
    a_atoms = r_atoms + [n - 1]  # make this (D, C, X)
    return zmat, coords, n, r_atoms, a_atoms, specific_last_d_atom


def get_parameter_from_atom_indices(
    zmat: dict,
    indices: Union[list, tuple],
    xyz_indexed: bool = True,
) -> Union[str, tuple, list]:
    """
    Get the zmat parameter from the atom indices.
    If indices are of length two, three, or four, an R, A, or D parameter is returned, respectively.

    If a requested parameter represents an angle split by a dummy atom,
    combine the two dummy angles to get the original angle.
    In this case, a list of the two corresponding parameters will be returned.
    """
    if not isinstance(indices, (list, tuple)):
        raise TypeError(f"indices must be a list, got {indices} which is a {type(indices)}")
    if len(indices) not in [2, 3, 4]:
        raise ZMatError(
            f"indices must be of length 2, 3, or 4, got {indices} (length {len(indices)}."
        )
    if xyz_indexed:
        if any(index not in list(zmat["map"].values()) for index in indices):
            raise ZMatError(
                f"Not all indices ({indices}) are in the zmat map values ({list(zmat['map'].values())})."
            )
        indices = [key_by_val(zmat["map"], index) for index in indices]
    if any(index not in list(zmat["map"].keys()) for index in indices):
        raise ZMatError(
            f"Not all indices ({indices}) are in the zmat map keys ({list(zmat['map'].keys())})."
        )
    key = "_".join([KEY_FROM_LEN[len(indices)]] + [str(index) for index in indices])
    if key in list(zmat["vars"].keys()):
        return key

    key = KEY_FROM_LEN[len(indices)]
    for var in zmat["vars"].keys():
        if var[0] == key and tuple(indices) in list(get_atom_indices_from_zmat_parameter(var)):
            return var

    var1, var2 = None, None
    if len(indices) == 3:
        dummy_indices = [
            str(key)
            for key, val in zmat["map"].items()
            if isinstance(val, str) and "X" in val
        ]
        param1 = "AX_{0}_{1}_{2}"
        all_parameters = list(zmat["vars"].keys())
        for dummy_str_index in dummy_indices:
            var1 = (
                param1.format(indices[0], indices[1], dummy_str_index)
                if param1.format(indices[0], indices[1], dummy_str_index) in all_parameters
                and var1 is None
                else var1
            )
            var1 = (
                param1.format(dummy_str_index, indices[1], indices[0])
                if param1.format(dummy_str_index, indices[1], indices[0]) in all_parameters
                and var1 is None
                else var1
            )
            if var1 is not None:
                param2a = f"AX_{dummy_str_index}_{indices[1]}_{indices[2]}"
                param2b = f"AX_{indices[2]}_{indices[1]}_{dummy_str_index}"
                var2 = param2a if param2a in all_parameters and var2 is None else var2
                var2 = param2b if param2b in all_parameters and var2 is None else var2
                if var2 is not None:
                    break
                var1 = None
    if var1 is not None and var2 is not None:
        return [var1, var2]
    raise ZMatError(f"Could not find a key corresponding to {key} {indices}.")


def get_atom_indices_from_zmat_parameter(param: str) -> tuple:
    """
    Get the atom indices from a zmat parameter.

    Examples:
        'R_0_2' --> ((0, 2),)
        'A_0_1_2' --> ((0, 1, 2),) corresponding to angle 0-1-2
        'D_0_1_2_4' --> ((0, 1, 2, 4),)
        'R_0|0_3|4' --> ((0, 3), (0, 4))
        'A_0|0|0_1|1|1_2|3|4' --> ((0, 1, 2), (0, 1, 3), (0, 1, 4)) corresponding to angles 0-1-2, 0-1-3, and 0-1-4
        'D_0|0|0_1|1|1_2|3|4_5|6|9' --> ((0, 1, 2, 5), (0, 1, 3, 6), (0, 1, 4, 9))
        'RX_0_2' --> ((0, 2),)

    Args:
        param (str): The zmat parameter.

    Returns:
        tuple: Entries are tuples of indices, each describing R, A, or D parameters.
               The tuple entries for R, A, and D types are of lengths 2, 3, and 4, respectively.
               The number of tuple entries depends on the number of consolidated parameters.
    """
    result, index_groups = list(), list()
    splits = param.split('_')[1:]  # exclude the type char ('R', 'A', or 'D')
    for split in splits:
        index_groups.append(split.split('|'))
    for i in range(len(index_groups[0])):
        result.append(tuple(int(index_group[i]) for index_group in index_groups))
    return tuple(result)


def xyz_to_x_y_z(xyz_dict: dict) -> Optional[Tuple[tuple, tuple, tuple]]:
    """
    Get the X, Y, and Z coordinates separately from the ARC xyz dictionary format.

    Args:
        xyz_dict (dict): The ARC xyz format.

    Returns: Optional[Tuple[tuple, tuple, tuple]]
        The X coordinates, the Y coordinates, the Z coordinates.
    """
    if xyz_dict is None:
        return None
    x, y, z = tuple(), tuple(), tuple()
    for coord in xyz_dict['coords']:
        x += (coord[0],)
        y += (coord[1],)
        z += (coord[2],)
    return x, y, z


def get_vector(pivot: int,
               anchor: int,
               xyz: dict,
               ) -> list:
    """
    Get a vector between two atoms in the molecule (pointing from pivot to anchor).

    Args:
        pivot (int): The 0-index of the pivotal atom around which groups are to be translated.
        anchor (int): The 0-index of an additional atom in the molecule.
        xyz (dict): The 3D coordinates of the molecule with the same atom order as in mol.

    Returns: list
         A vector pointing from the pivotal atom towards the anchor atom.
    """
    x, y, z = xyz_to_x_y_z(xyz)
    dx = x[anchor] - x[pivot]
    dy = y[anchor] - y[pivot]
    dz = z[anchor] - z[pivot]
    return [dx, dy, dz]



def get_vector_length(v: List[float]) -> float:
    """
    Get the length of an ND vector

    Args:
        v (list): The vector.

    Returns: float
        The vector's length.
    """
    return float(np.dot(v, v) ** 0.5)


def _add_nth_atom_to_coords(zmat: dict,
                            coords: list,
                            i: int,
                            coords_to_skip: Optional[list] = None,
                            ) -> list:
    """
    Add the n-th atom to the coords (n >= 0).

    Args:
        zmat (dict): The zmat.
        coords (list): The coordinates to be updated (not the entire xyz dict).
        i (int): The atom number in the zmat to be added to the coords (0-indexed)
        coords_to_skip (list, optional): Entries are indices to skip.

    Returns:
        list: The updated coords.
    """
    coords_to_skip = coords_to_skip or list()
    if i == 0:
        # Add the 1st atom.
        coords.append((0.0, 0.0, 0.0))  # atom A is placed at the origin
    elif i == 1:
        # Add the 2nd atom.
        r_key = zmat['coords'][i][0]
        coords.append((0.0, 0.0, zmat['vars'][r_key]))  # atom B is placed on axis Z, distant by the AB bond length
    elif i == 2:
        # Add the 3rd atom (atom "C").
        r_key, a_key = zmat['coords'][i][0], zmat['coords'][i][1]
        bc_length = zmat['vars'][r_key]
        alpha = zmat['vars'][a_key]
        alpha = math.radians(alpha if alpha < 180 else 360 - alpha)
        b_index = [indices for indices in get_atom_indices_from_zmat_parameter(r_key) if indices[0] == i][0][1]
        b_z = coords[b_index][2]
        c_y = bc_length * math.sin(alpha)
        r"""
        We differentiate between two cases for c_z:
        Either atom A is at the origin (case 1), or atom B is at the origin (case 2).
        One of them has to be at the origin (0, 0, 0), since we're adding the 3rd atom (so either A or B were 1st).
        
         y
         ^                    C                         C
         |           (1)       \        or     (2)     /
         L__ > z           A -- B                    B -- A
        
        In case 1, we need to deduct len(B-C) from the z coordinate of atom B,
        but in case 2 we need to take the positive value of len(B-C).
        The above is also true if alpha(A-B-C) is > 90 degrees.
        """
        c_z = b_z - bc_length * math.cos(alpha) if b_z else bc_length * math.cos(alpha)
        coords.append((0.0, c_y, c_z))
    elif i not in coords_to_skip:
        d_indices = [indices for indices in get_atom_indices_from_zmat_parameter(zmat['coords'][i][2])
                     if indices[0] == i][0]
        a_index, b_index, c_index = d_indices[3], d_indices[2], d_indices[1]
        # Atoms B and C aren't necessarily connected in the zmat, calculate from coords.
        bc_length = get_vector_length([coords[c_index][0] - coords[b_index][0],
                                       coords[c_index][1] - coords[b_index][1],
                                       coords[c_index][2] - coords[b_index][2]])
        cd_length = zmat['vars'][zmat['coords'][i][0]]
        bcd_angle = math.radians(zmat['vars'][zmat['coords'][i][1]])
        abcd_dihedral = math.radians(zmat['vars'][zmat['coords'][i][2]])
        # A vector pointing from atom A to atom B:
        ab = [(coords[b_index][0] - coords[a_index][0]),
              (coords[b_index][1] - coords[a_index][1]),
              (coords[b_index][2] - coords[a_index][2])]
        # A normalized vector pointing from atom B to atom C:
        ubc = [(coords[c_index][0] - coords[b_index][0]) / bc_length,
               (coords[c_index][1] - coords[b_index][1]) / bc_length,
               (coords[c_index][2] - coords[b_index][2]) / bc_length]
        n = np.cross(ab, ubc)
        un = n / get_vector_length(n)
        un_cross_ubc = np.cross(un, ubc)

        # The transformation matrix:
        m = np.array([[ubc[0], un_cross_ubc[0], un[0]],
                      [ubc[1], un_cross_ubc[1], un[1]],
                      [ubc[2], un_cross_ubc[2], un[2]]], np.float64)

        # Place atom D in a default coordinate system.
        d = np.array([- cd_length * math.cos(bcd_angle),
                      cd_length * math.sin(bcd_angle) * math.cos(abcd_dihedral),
                      cd_length * math.sin(bcd_angle) * math.sin(abcd_dihedral)])
        d = m.dot(d)  # Rotate the coordinate system into the reference frame of orientation defined by A, B, C.
        # Add the coordinates of atom C to the resulting atom D:
        coords.append((float(d[0] + coords[c_index][0]), float(d[1] + coords[c_index][1]), float(d[2] + coords[c_index][2])))
    return coords


def remap_zmat_parameter(
    param: Optional[str],
    index_map: Dict[int, int],
) -> Optional[str]:
    if param is None:
        return None

    prefix = param.split("_", 1)[0]
    index_groups = get_atom_indices_from_zmat_parameter(param)
    mapped = [tuple(index_map[idx] for idx in group) for group in index_groups]

    if len(mapped) == 1:
        return prefix + "_" + "_".join(str(idx) for idx in mapped[0])

    cols = []
    for i in range(len(mapped[0])):
        cols.append("|".join(str(group[i]) for group in mapped))
    return prefix + "_" + "_".join(cols)


def zmat_to_xyz(zmat: dict, keep_dummy: bool = False) -> Dict[str, tuple]:
    coords, symbols = zmat_to_coords(zmat=zmat, keep_dummy=keep_dummy)
    return {"symbols": tuple(symbols), "coords": tuple(coords)}


def _is_atom_in_linear_angle_rmg(i: int, xyz: Optional[dict], mol: "RMG_Molecule", tol: float = 0.9) -> bool:
    if not xyz:
        return False
    atom_to_index = {atom: idx for idx, atom in enumerate(mol.atoms)}
    for b in range(len(mol.atoms)):
        b_neighbors = [atom_to_index[nbr] for nbr in mol.atoms[b].edges]
        for a in b_neighbors:
            for c in b_neighbors:
                if a >= c:
                    continue
                if i not in (a, b, c):
                    continue
                angle = calculate_param(coords=xyz["coords"], atoms=[a, b, c])
                if is_angle_linear(angle, tolerance=tol):
                    return True
    return False


def _get_atom_order_from_rmg(
    mol: "RMG_Molecule",
    fragment: Optional[List[int]] = None,
    constraints_dict: Optional[Dict[str, List[tuple]]] = None,
    xyz: Optional[dict] = None,
) -> List[int]:
    fragment = set(fragment or range(len(mol.atoms)))
    constraints = constraints_dict or {}
    if not constraints:
        active = None
    else:
        key, tpl_list = next(iter(constraints.items()))
        active = (key, tpl_list[0])

    atom_to_index = {atom: idx for idx, atom in enumerate(mol.atoms)}

    constrained_set = set()
    if active:
        key, tpl = active
        constrained_set.update(tpl)
        if key in ("R_group", "A_group", "D_group"):
            root, anchor = tpl[0], tpl[1]
            seen = {root}
            queue = [root]
            while queue:
                i = queue.pop(0)
                for nbr in mol.atoms[i].edges:
                    j = atom_to_index[nbr]
                    if j in fragment and j not in seen and not (i == root and j == anchor):
                        seen.add(j)
                        queue.append(j)
            constrained_set |= seen
        elif key == "D_groups":
            for root in tpl[:2]:
                seen = {root}
                queue = [root]
                while queue:
                    i = queue.pop(0)
                    for nbr in mol.atoms[i].edges:
                        j = atom_to_index[nbr]
                        if j in fragment and j not in seen:
                            seen.add(j)
                            queue.append(j)
                constrained_set |= seen

    def _is_hydrogen(idx: int) -> bool:
        return mol.atoms[idx].is_hydrogen()

    def _heavy_neighbor_count(idx: int) -> int:
        return sum(
            1
            for nbr in mol.atoms[idx].edges
            if (atom_to_index[nbr] in fragment and not nbr.is_hydrogen())
        )

    def find_start(avoid_linear: bool = True) -> int:
        for atom in mol.atoms:
            i = atom_to_index[atom]
            if (
                i in fragment
                and not _is_hydrogen(i)
                and i not in constrained_set
                and (not avoid_linear or not _is_atom_in_linear_angle_rmg(i=i, xyz=xyz, mol=mol))
                and _heavy_neighbor_count(i) <= 1
            ):
                return i
        for atom in mol.atoms:
            i = atom_to_index[atom]
            if (
                i in fragment
                and not _is_hydrogen(i)
                and i not in constrained_set
                and (not avoid_linear or not _is_atom_in_linear_angle_rmg(i=i, xyz=xyz, mol=mol))
            ):
                return i
        for atom in mol.atoms:
            i = atom_to_index[atom]
            if i in fragment and not _is_hydrogen(i):
                return i
        return next(iter(fragment))

    start = find_start(avoid_linear=True)

    visited = set()
    base_heavies: List[int] = []
    queue = [start]
    while queue:
        i = queue.pop(0)
        if i in visited or i not in fragment:
            continue
        visited.add(i)
        if not _is_hydrogen(i):
            base_heavies.append(i)
            for nbr in mol.atoms[i].edges:
                j = atom_to_index[nbr]
                if j not in visited and not nbr.is_hydrogen() and j in fragment:
                    queue.append(j)

    base_hydrogens = [i for i in range(len(mol.atoms)) if i in fragment and _is_hydrogen(i)]

    heavy_uncon = [i for i in base_heavies if i not in constrained_set]
    h_uncon = [i for i in base_hydrogens if i not in constrained_set]

    tail: List[int] = []
    if active:
        key, tpl = active
        if key == "R_atom":
            tail = [tpl[1], tpl[0]]
        elif key == "A_atom":
            tail = [tpl[2], tpl[1], tpl[0]]
        elif key == "D_atom":
            tail = [tpl[3], tpl[2], tpl[1], tpl[0]]
        elif key == "R_group":
            root, anchor = tpl
            seen = {root}
            queue = [root]
            while queue:
                i = queue.pop(0)
                for nbr in mol.atoms[i].edges:
                    j = atom_to_index[nbr]
                    if j in fragment and j not in seen and not (i == root and j == anchor):
                        seen.add(j)
                        queue.append(j)
            group = [root] + [x for x in seen if x != root]
            tail = [anchor, root] + [x for x in group if x != root]
        elif key == "A_group":
            root, ref1, ref2 = tpl
            seen = {root}
            queue = [root]
            while queue:
                i = queue.pop(0)
                for nbr in mol.atoms[i].edges:
                    j = atom_to_index[nbr]
                    if j in fragment and j not in seen and not (i == root and j in (ref1, ref2)):
                        seen.add(j)
                        queue.append(j)
            group = [root] + [x for x in seen if x != root]
            tail = [ref2, ref1, root] + [x for x in group if x != root]
        elif key == "D_group":
            root, ref1, ref2, ref3 = tpl
            seen = {root}
            queue = [root]
            while queue:
                i = queue.pop(0)
                for nbr in mol.atoms[i].edges:
                    j = atom_to_index[nbr]
                    if j in fragment and j not in seen and not (i == root and j == ref1):
                        seen.add(j)
                        queue.append(j)
            group = [root] + [x for x in seen if x != root]
            tail = [ref3, ref2, ref1, root] + [x for x in group if x != root]
        elif key == "D_groups":
            pivot2, pivot3, ref1, ref2 = tpl[:4]
            seen_all = set()
            for root in (pivot2, pivot3):
                seen = {root}
                queue = [root]
                while queue:
                    i = queue.pop(0)
                    for nbr in mol.atoms[i].edges:
                        j = atom_to_index[nbr]
                        if j in fragment and j not in seen:
                            seen.add(j)
                            queue.append(j)
                seen_all |= seen
            tail = [ref2, ref1, pivot3, pivot2] + [x for x in seen_all if x not in (pivot2, pivot3)]

    atom_order = heavy_uncon + h_uncon + tail
    seen = set()
    ordered: List[int] = []
    for i in atom_order:
        if i in fragment and i not in seen:
            ordered.append(i)
            seen.add(i)
    for i in fragment:
        if i not in seen:
            ordered.append(i)
    return ordered


def get_rmg_atom_order_from_rdkit(
    rdmol: Chem.Mol,
    constraints_dict: Optional[Dict[str, List[tuple]]] = None,
    fragments: Optional[List[List[int]]] = None,
    xyz: Optional[dict] = None,
) -> Optional[List[int]]:
    """
    Return a RDKit-indexed atom order that follows RMG's sorted-atom ordering.
    Returns None if RMG is unavailable or mapping fails.
    """
    if RMG_Molecule is None or rmg_converter is None:
        return None
    if rdmol is None:
        return None

    rmg_mol = RMG_Molecule()
    rmg_converter.from_rdkit_mol(rmg_mol, rdmol)

    rmg_rdkit, rmg_to_rdk = rmg_converter.to_rdkit_mol(
        rmg_mol, remove_h=False, return_mapping=True, save_order=True
    )
    rmg_idx_to_rdk = {
        i: rmg_to_rdk[atom]
        for i, atom in enumerate(rmg_mol.vertices)
        if atom in rmg_to_rdk
    }

    match = rdmol.GetSubstructMatch(rmg_rdkit)
    if not match:
        inverse = rmg_rdkit.GetSubstructMatch(rdmol)
        if inverse:
            remapped = [None] * len(rmg_rdkit.GetAtoms())
            for orig_idx, rmg_idx in enumerate(inverse):
                if rmg_idx < len(remapped):
                    remapped[rmg_idx] = orig_idx
            match = tuple(i for i in remapped if i is not None)
        else:
            return None

    rmg_to_orig = {
        rmg_idx: match[rmg_idx_to_rdk[rmg_idx]]
        for rmg_idx in rmg_idx_to_rdk
        if rmg_idx_to_rdk[rmg_idx] < len(match)
    }
    if not rmg_to_orig:
        return None

    frag_list = fragments or [list(range(len(rmg_mol.vertices)))]
    rmg_order = []
    for frag in frag_list:
        rmg_order.extend(_get_atom_order_from_rmg(rmg_mol, fragment=frag, constraints_dict=constraints_dict, xyz=xyz))

    return [rmg_to_orig[i] for i in rmg_order if i in rmg_to_orig]


def get_rmg_connectivity_from_rdkit(rdmol: Chem.Mol) -> Optional[Dict[int, List[int]]]:
    """
    Return RDKit-indexed connectivity, but with neighbor ordering derived from RMG's atom order.
    """
    if RMG_Molecule is None or rmg_converter is None:
        return None
    if rdmol is None:
        return None

    rmg_mol = RMG_Molecule()
    rmg_converter.from_rdkit_mol(rmg_mol, rdmol)

    rmg_rdkit, rmg_to_rdk = rmg_converter.to_rdkit_mol(
        rmg_mol, remove_h=False, return_mapping=True, save_order=True
    )
    rmg_idx_to_rdk = {
        i: rmg_to_rdk[atom]
        for i, atom in enumerate(rmg_mol.vertices)
        if atom in rmg_to_rdk
    }

    match = rdmol.GetSubstructMatch(rmg_rdkit)
    if not match:
        inverse = rmg_rdkit.GetSubstructMatch(rdmol)
        if inverse:
            remapped = [None] * len(rmg_rdkit.GetAtoms())
            for orig_idx, rmg_idx in enumerate(inverse):
                if rmg_idx < len(remapped):
                    remapped[rmg_idx] = orig_idx
            match = tuple(i for i in remapped if i is not None)
        else:
            return None

    rmg_to_orig = {
        rmg_idx: match[rmg_idx_to_rdk[rmg_idx]]
        for rmg_idx in rmg_idx_to_rdk
        if rmg_idx_to_rdk[rmg_idx] < len(match)
    }
    if not rmg_to_orig:
        return None

    atom_to_index = {atom: idx for idx, atom in enumerate(rmg_mol.atoms)}
    adj: Dict[int, List[int]] = {i: [] for i in range(rdmol.GetNumAtoms())}
    for rmg_atom in rmg_mol.atoms:
        rmg_idx = atom_to_index[rmg_atom]
        if rmg_idx not in rmg_to_orig:
            continue
        rd_idx = rmg_to_orig[rmg_idx]
        for nbr in rmg_atom.edges:
            nbr_idx = atom_to_index[nbr]
            if nbr_idx in rmg_to_orig:
                adj[rd_idx].append(rmg_to_orig[nbr_idx])

    return adj


def _pick_non_linear_angle_ref(
    zmat: Dict[str, Union[dict, tuple]],
    coords: Union[list, tuple],
    atom_index: int,
    r_atoms: List[int],                 # [n, C] style? in your code r_atoms is [n, C_z]
    cand_z_indices: List[int],          # candidates in zmat-index space
    tol: float,
) -> Optional[int]:
    """
    Pick a zmat index B from cand_z_indices such that angle(n - C - B) is not linear.
    Here C is r_atoms[1].
    """
    if r_atoms is None or len(r_atoms) != 2:
        return None

    C_z = r_atoms[1]
    C_xyz = zmat["map"][C_z]

    for B_z in cand_z_indices:
        if B_z in (r_atoms[0], C_z):
            continue
        if B_z not in zmat["map"]:
            continue
        if B_z < len(zmat["symbols"]) and is_dummy(zmat, B_z):
            continue

        B_xyz = zmat["map"][B_z]
        if isinstance(B_xyz, str) or isinstance(C_xyz, str):
            continue

        try:
            ang = calculate_param(coords=coords, atoms=[atom_index, C_xyz, B_xyz])
        except VectorsError:
            continue

        if not is_angle_linear(ang, tolerance=tol):
            return B_z

    return None


def determine_a_atoms(
    zmat: Dict[str, Union[dict, tuple]],
    coords: Union[list, tuple],
    connectivity: Optional[Dict[int, Iterable[int]]],
    r_atoms: Optional[List[int]],
    n: int,
    atom_index: int,
    a_constraint: Optional[Tuple[int, int, int]] = None,
    d_constraint: Optional[Tuple[int, int, int, int]] = None,
    a_constraint_type: Optional[str] = None,
    trivial_assignment: bool = False,
    fragments: Optional[List[List[int]]] = None,
) -> Optional[List[int]]:
    """
    Determine the atoms for defining the angle A.
    This should be in the form: [n, r_atoms[1], <some other atom already in the zmat>]
    """

    if r_atoms is not None and is_atom_in_new_fragment(
        atom_index=atom_index, zmat=zmat, fragments=fragments, skip_atoms=r_atoms
    ):
        connectivity = None
    if r_atoms is not None and len(r_atoms) != 2:
        raise ZMatError(f"r_atoms must be a list of length 2, got {r_atoms}")
    if len(zmat["coords"]) <= 1:
        a_atoms = None
    elif a_constraint is not None:
        if a_constraint_type not in ["A_atom", "A_group", None]:
            raise ZMatError(f'Got an invalid A constraint type "{a_constraint_type}" for {a_constraint}')
        a_atoms = [n] + [key_by_val(zmat["map"], atom) for atom in a_constraint[1:]]
    elif d_constraint is not None:
        a_atoms = [n] + [key_by_val(zmat["map"], atom) for atom in d_constraint[1:3]]
    elif connectivity is not None and r_atoms is not None:
        a_atoms = [atom for atom in r_atoms]
        c_xyz = zmat["map"][a_atoms[-1]]
        if isinstance(c_xyz, int):
            for atom in list(connectivity.get(c_xyz, [])) + list(connectivity.get(atom_index, [])):
                if atom in list(zmat["map"].values()):
                    zmat_index = key_by_val(zmat["map"], atom)
                    if atom != atom_index and zmat_index not in a_atoms and not is_dummy(zmat, zmat_index):
                        i = 0
                        atom_b, atom_c = atom, zmat["map"][r_atoms[1]]
                        while i < len(list(connectivity.keys())):
                            num_of_neighbors = len(list(connectivity[atom_b]))
                            if num_of_neighbors == 1:
                                break
                            if num_of_neighbors == 2:
                                b_neighbors = list(connectivity[atom_b])
                                atom_a = b_neighbors[0] if b_neighbors[0] != atom_c else b_neighbors[1]
                                zmat_index = key_by_val(zmat["map"], atom_b)
                            else:
                                zmat_index = key_by_val(zmat["map"], atom_b)
                                break
                            i += 1
                        a_atoms.append(zmat_index)
                        break
            if len(set(a_atoms)) != 3:
                trivial_assignment = True
        else:
            trivial_assignment = True
    else:
        trivial_assignment = True
        a_atoms = list()
    if trivial_assignment and isinstance(a_atoms, list) and len(a_atoms) != 3:
        a_atoms = [atom for atom in r_atoms] if r_atoms is not None else []
        for i in reversed(range(n)):
            zmat_index = i
            if i not in a_atoms and i in list(zmat["map"].keys()) and not is_dummy(zmat, i):
                zmat_index, j = i, n - 1
                atom_b, atom_c = zmat["map"][i], zmat["map"][r_atoms[1]]
                while j > 0:
                    atom_a = zmat["map"][j]
                    if j != i and atom_a not in [atom_b, atom_c] and (
                        j in list(zmat["map"].keys()) and not is_dummy(zmat, j) or j not in list(zmat["map"].keys())
                    ):
                        zmat_index = key_by_val(zmat["map"], atom_b)
                        a_atoms.append(zmat_index)
                        break
                    j -= 1
            if len(a_atoms) == 3:
                break
        if len(a_atoms) == 2 and zmat_index not in a_atoms:
            a_atoms.append(zmat_index)
    if a_atoms is not None and any(a_atom not in list(zmat["map"].keys()) for a_atom in a_atoms[1:]):
        raise ZMatError(
            f"The reference A atom in {a_atoms} for the index atom {atom_index} has not been "
            f"added to the zmat yet. Added atoms are (zmat index: xyz index): {zmat['map']}."
        )
    if a_atoms is not None and len(set(a_atoms)) != 3:
        for i in reversed(range(len(coords))):
            if i not in a_atoms and i in list(zmat["map"].keys()) and not is_dummy(zmat, i):
                a_atoms.append(i)
                break
    if a_atoms is not None and len(set(a_atoms)) != 3:
        raise ZMatError(f"Could not come up with three unique a_atoms (a_atoms = {a_atoms}).")
    return a_atoms


def is_xyz_linear(xyz: Optional[dict]) -> Optional[bool]:
    """
    Determine whether the xyz coords represents a linear molecule.

    Args:
        xyz (dict): The xyz coordinates in dict format.

    Returns:
        bool: Whether the molecule is linear, ``True`` if it is.
    """
    if not xyz or 'coords' not in xyz or 'symbols' not in xyz:
        return None
    coordinates = np.array(xyz['coords'])
    n_atoms = len(coordinates)
    if n_atoms == 1:
        return False
    if n_atoms == 2:
        return True

    for i in range(1, n_atoms - 1):
        v1 = coordinates[i - 1] - coordinates[i]
        v2 = coordinates[i + 1] - coordinates[i]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            continue
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.degrees(np.arccos(cos_angle))
        if not is_angle_linear(angle, tolerance=0.1):
            return False
    return True


def _rdkit_atom_signature(rdmol, atom_idx: int) -> Tuple[Any, ...]:
    """
    A stable-ish atom 'type' signature from RDKit, to replace RMG atomtype.label for consolidation checks.
    """
    a = rdmol.GetAtomWithIdx(atom_idx)
    return (
        a.GetSymbol(),
        a.GetIsAromatic(),
        int(a.GetHybridization()),
        a.GetTotalDegree(),
        a.GetTotalValence(),
        a.GetFormalCharge(),
        a.IsInRing(),
    )

def consolidate_zmat(
    zmat: dict,
    mol: Optional["Molecule"] = None,   # RMG Molecule, optional
    rdmol: Optional[object] = None,     # RDKit Mol, optional
    consolidation_tols: Optional[dict] = None,
) -> dict:
    """
    Consolidate (almost) identical vars in the zmat.

    If `mol` is given, uses RMG atom types.
    Else if `rdmol` is given, uses an RDKit atom signature.
    Else falls back to element symbols.
    """
    consolidation_tols = consolidation_tols or dict()

    if "R" not in consolidation_tols:
        consolidation_tols["R"] = DEFAULT_CONSOLIDATION_R_TOL
    if "A" not in consolidation_tols:
        consolidation_tols["A"] = DEFAULT_CONSOLIDATION_A_TOL
    if "D" not in consolidation_tols:
        consolidation_tols["D"] = DEFAULT_CONSOLIDATION_D_TOL

    zmat["coords"] = list(zmat["coords"])

    keys_to_consolidate1 = {"R": [], "A": [], "D": []}
    keys_to_consolidate2 = {"R": [], "A": [], "D": []}

    # 1) find numeric-close keys
    for i, key1 in enumerate(zmat["vars"].keys()):
        if key1 is None:
            continue
        if any(key1 in group for group in keys_to_consolidate1[key1[0]]):
            continue

        dup_keys = []
        for j, key2 in enumerate(zmat["vars"].keys()):
            if j <= i or key2 is None:
                continue
            if key1[0] != key2[0]:
                continue
            if abs(zmat["vars"][key1] - zmat["vars"][key2]) < consolidation_tols[key1[0]]:
                for key in (key1, key2):
                    if key not in dup_keys:
                        dup_keys.append(key)

        if dup_keys:
            appended = False
            for dup_key in dup_keys:
                for g in range(len(keys_to_consolidate1[key1[0]])):
                    if dup_key in keys_to_consolidate1[key1[0]][g]:
                        keys_to_consolidate1[key1[0]][g] = sorted(
                            list(set(keys_to_consolidate1[key1[0]][g] + dup_keys))
                        )
                        appended = True
            if not appended:
                keys_to_consolidate1[key1[0]].append(dup_keys)

    # 2) split by "atom identity" (RMG atomtype OR RDKit signature OR symbol)
    for key_type in ["R", "A", "D"]:
        for keys in keys_to_consolidate1[key_type]:
            atoms_dict = {}

            for key in keys:
                indices = [int(idx) for idx in key.split("_")[1:]]

                if any(zmat["symbols"][idx] == "X" for idx in indices):
                    # dummy => always consolidate
                    atoms_dict[key] = (key_type, "X")
                    continue

                if mol is not None:
                    atoms_dict[key] = tuple(
                        mol.atoms[zmat["map"][idx]].atomtype.label for idx in indices
                    )
                elif rdmol is not None:
                    atoms_dict[key] = tuple(
                        _rdkit_atom_signature(rdmol, zmat["map"][idx]) for idx in indices
                    )
                else:
                    atoms_dict[key] = tuple(zmat["symbols"][idx] for idx in indices)

            atoms_sets = list(set(atoms_dict.values()))

            # drop reverse duplicates (e.g. (C,H) same as (H,C)), but keep symmetric tuples
            indices_to_pop = []
            for i, atoms_set in enumerate(atoms_sets):
                rev = tuple(reversed(atoms_set))
                if rev in atoms_sets and rev not in [atoms_sets[j] for j in indices_to_pop] and atoms_set != rev:
                    indices_to_pop.append(i)
            for i in reversed(range(len(atoms_sets))):
                if i in indices_to_pop:
                    atoms_sets.pop(i)

            for atoms_tuple in atoms_sets:
                keys_to_consolidate2[key_type].append(
                    [k for k, v in atoms_dict.items() if v == atoms_tuple or tuple(reversed(v)) == atoms_tuple]
                )

    # 3) consolidate
    for key_type in ["R", "A", "D"]:
        for keys in keys_to_consolidate2[key_type]:
            # new key name
            indices = []
            for pos in range(len(keys[0].split("_")[1:])):
                indices.append([key.split("_")[1:][pos] for key in keys])
            new_indices = ["|".join(str(x) for x in col) for col in indices]

            if any(zmat["symbols"][int(idx)] == "X" for col in indices for idx in col):
                new_key = "_".join([key_type + "X"] + new_indices)
            else:
                new_key = "_".join([key_type] + new_indices)

            # replace in coords
            for i in range(len(zmat["coords"])):
                if any(coord in keys for coord in zmat["coords"][i]):
                    zmat["coords"][i] = tuple(new_key if coord in keys else coord for coord in zmat["coords"][i])

            # replace in vars (average)
            if all(k in zmat["vars"] for k in keys):
                vals = [zmat["vars"][k] for k in keys]
                new_value = sum(vals) / len(vals)
                for k in keys:
                    del zmat["vars"][k]
                zmat["vars"][new_key] = new_value
            else:
                # keep your original "merge into existing consolidated key" behavior
                found_key = False
                keys_as_indices = [get_atom_indices_from_zmat_parameter(k)[0] for k in keys]
                for variable in list(zmat["vars"].keys()):
                    if variable[0] != key_type:
                        continue
                    var_indices = get_atom_indices_from_zmat_parameter(variable)
                    if any(key_idx in var_indices for key_idx in keys_as_indices):
                        found_key = True
                        for key_idx in keys_as_indices:
                            if key_idx not in var_indices:
                                var_indices += (key_idx,)
                        new_consolidated_key = key_type
                        for p in range(len(var_indices[0])):
                            new_consolidated_key += "_" + "|".join(str(v[p]) for v in var_indices)
                        zmat["vars"][new_consolidated_key] = zmat["vars"][variable]
                        del zmat["vars"][variable]
                        break

                if found_key:
                    for k in keys:
                        if k in zmat["vars"]:
                            del zmat["vars"][k]
                else:
                    raise ZMatError("Could not consolidate zmat")

                # NOTE: new_value must be defined in this branch too; choose existing value
                zmat["vars"][new_key] = zmat["vars"][new_consolidated_key]

    zmat["coords"] = tuple(zmat["coords"])
    return zmat
