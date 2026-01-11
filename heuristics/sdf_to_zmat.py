from typing import Dict, List, Optional, Tuple, Set, Iterable, Any, Union

import re
from rdkit import Chem

from .utils import (
    ZMatError,
    check_atom_r_constraints,
    check_atom_d_constraints,
    check_atom_a_constraints,
    determine_d_atoms,
    determine_r_atoms,
    determine_a_atoms,
    is_angle_linear,
    calculate_param,
    TOL_180,
    add_dummy_atom,
    update_zmat_with_new_atom,
    consolidate_zmat,
    get_rmg_atom_order_from_rdkit,
    get_rmg_connectivity_from_rdkit,
)


def get_connectivity_from_rdkit(rdmol: Chem.Mol) -> Dict[int, List[int]]:
    """
    Docstring for get_connectivity_from_rdkit
    
    :param rdmol: Description
    :type rdmol: Chem.Mol
    :return: Description
    :rtype: Dict[int, Set[int]]
    """
    n = rdmol.GetNumAtoms()
    adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for b in rdmol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        adj[i].add(j)
        adj[j].add(i)
    # Return sorted neighbor lists for deterministic traversal.
    return {i: sorted(neigh) for i, neigh in adj.items()}


def connected_component(
    adj: Dict[int, Iterable[int]],
    start: int,
    block: Optional[Set[int]] = None,
) -> Set[int]:
    """
    Docstring for connect_component
    
    :param adj: Description
    :type adj: Dict[int, Set[int]]
    :param start: Description
    :type start: int
    :param block: Description
    :type block: Optional[Set[int]]
    :return: Description
    :rtype: Set[int]
    """
    blocked = block or set()
    if start in blocked:
        return set()
    
    seen = set()
    stack = [start]
    while stack:
        u = stack.pop()
        if u in seen or u in blocked:
            continue
        seen.add(u)
        for v in adj[u]:
            if v not in seen and v not in blocked:
                stack.append(v)
    return seen


def infer_fragments_from_connectivity(adj: Dict[int, Set[int]], n_atoms: int) -> List[List[int]]:
    """
    Docstring for infer_fragments_from_connectivity
    
    :param adj: Description
    :type adj: Dict[int, Set[int]]
    :param n_atoms: Description
    :type n_atoms: int
    :return: Description
    :rtype: List[List[int]]
    """
    unassigned = set(range(n_atoms))
    frags: List[List[int]] = []
    while unassigned:
        seed = next(iter(unassigned))
        comp = connected_component(adj, seed)
        frags.append(sorted(comp))
        unassigned -= comp
    return frags


def get_atom_order(
    xyz: Optional[Dict[str, tuple]] = None,
    rdmol: Optional[Chem.Mol] = None,
    fragments: Optional[List[List[int]]] = None,
    constraints_dict: Optional[Dict[str, List[tuple]]] = None,
) -> List[int]:
    if rdmol is None and xyz is None:
        raise ValueError("Either rdmol or xyz must be provided.")

    if not fragments:
        if rdmol is not None:
            fragments = [list(range(rdmol.GetNumAtoms()))]
        else:
            fragments = [list(range(len(xyz["symbols"])))]

    atom_order: List[int] = []
    if rdmol is not None:
        for fragment in fragments:
            sequence = get_atom_order_from_rdkit(
                rdmol=rdmol,
                fragment=fragment,
                constraints_dict=constraints_dict,
                xyz=xyz,
            )
            for i in sequence:
                if i not in atom_order:
                    atom_order.append(i)
    else:
        for fragment in fragments:
            sequence = get_atom_order_from_xyz(xyz=xyz, fragment=fragment)
            for i in sequence:
                if i not in atom_order:
                    atom_order.append(i)
    return atom_order


def get_atom_order_from_rdkit(
    rdmol: Chem.Mol,
    fragment: Optional[List[int]] = None,
    constraints_dict: Optional[Dict[str, List[tuple]]] = None,
    xyz: Optional[dict] = None,
) -> List[int]:
    fragment = list(fragment or range(rdmol.GetNumAtoms()))
    fragment_set = set(fragment)
    constraints = constraints_dict or {}

    if not constraints:
        active = None
    else:
        key, tpl_list = next(iter(constraints.items()))
        active = (key, tpl_list[0])

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
                for nbr in rdmol.GetAtomWithIdx(i).GetNeighbors():
                    j = nbr.GetIdx()
                    if j in fragment_set and j not in seen and not (i == root and j == anchor):
                        seen.add(j)
                        queue.append(j)
            constrained_set |= seen
        elif key == "D_groups":
            for root in tpl[:2]:
                seen = {root}
                queue = [root]
                while queue:
                    i = queue.pop(0)
                    for nbr in rdmol.GetAtomWithIdx(i).GetNeighbors():
                        j = nbr.GetIdx()
                        if j in fragment_set and j not in seen:
                            seen.add(j)
                            queue.append(j)
                constrained_set |= seen

    def _is_hydrogen(idx: int) -> bool:
        return rdmol.GetAtomWithIdx(idx).GetSymbol() == "H"

    def _heavy_neighbor_count(idx: int) -> int:
        return sum(
            1
            for nbr in rdmol.GetAtomWithIdx(idx).GetNeighbors()
            if nbr.GetIdx() in fragment_set and nbr.GetSymbol() != "H"
        )

    def find_start(avoid_linear: bool = True) -> int:
        for atom in rdmol.GetAtoms():
            i = atom.GetIdx()
            if (
                i in fragment_set
                and not _is_hydrogen(i)
                and i not in constrained_set
                and (not avoid_linear or not is_atom_in_linear_angle(i=i, xyz=xyz, rdmol=rdmol))
                and _heavy_neighbor_count(i) <= 1
            ):
                return i
        for atom in rdmol.GetAtoms():
            i = atom.GetIdx()
            if (
                i in fragment_set
                and not _is_hydrogen(i)
                and i not in constrained_set
                and (not avoid_linear or not is_atom_in_linear_angle(i=i, xyz=xyz, rdmol=rdmol))
            ):
                return i
        for atom in rdmol.GetAtoms():
            i = atom.GetIdx()
            if i in fragment_set and not _is_hydrogen(i):
                return i
        return fragment[0]

    start = find_start(avoid_linear=True)

    visited = set()
    base_heavies: List[int] = []
    queue = [start]
    while queue:
        i = queue.pop(0)
        if i in visited or i not in fragment_set:
            continue
        visited.add(i)
        if not _is_hydrogen(i):
            base_heavies.append(i)
            for nbr in rdmol.GetAtomWithIdx(i).GetNeighbors():
                j = nbr.GetIdx()
                if j not in visited and not _is_hydrogen(j) and j in fragment_set:
                    queue.append(j)

    base_hydrogens = [i for i in fragment if _is_hydrogen(i)]

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
                for nbr in rdmol.GetAtomWithIdx(i).GetNeighbors():
                    j = nbr.GetIdx()
                    if j in fragment_set and j not in seen and not (i == root and j == anchor):
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
                for nbr in rdmol.GetAtomWithIdx(i).GetNeighbors():
                    j = nbr.GetIdx()
                    if j in fragment_set and j not in seen and not (i == root and j in (ref1, ref2)):
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
                for nbr in rdmol.GetAtomWithIdx(i).GetNeighbors():
                    j = nbr.GetIdx()
                    if j in fragment_set and j not in seen and not (i == root and j == ref1):
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
                    for nbr in rdmol.GetAtomWithIdx(i).GetNeighbors():
                        j = nbr.GetIdx()
                        if j in fragment_set and j not in seen:
                            seen.add(j)
                            queue.append(j)
                seen_all |= seen
            tail = [ref2, ref1, pivot3, pivot2] + [x for x in seen_all if x not in (pivot2, pivot3)]

    atom_order = heavy_uncon + h_uncon + tail
    seen = set()
    ordered: List[int] = []
    for i in atom_order:
        if i in fragment_set and i not in seen:
            ordered.append(i)
            seen.add(i)
    for i in fragment:
        if i not in seen:
            ordered.append(i)
    return ordered


def is_atom_in_linear_angle(i: int, xyz: Optional[dict], rdmol: Chem.Mol, tol: float = 0.9) -> bool:
    if not xyz:
        return False
    for b in range(rdmol.GetNumAtoms()):
        b_neighbors = [nbr.GetIdx() for nbr in rdmol.GetAtomWithIdx(b).GetNeighbors()]
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


def get_atom_order_from_xyz(xyz: Dict[str, tuple], fragment: Optional[List[int]] = None) -> List[int]:
    fragment = fragment or list(range(len(xyz["symbols"])))
    atom_order: List[int] = []
    hydrogens: List[int] = []
    for i, symbol in enumerate(xyz["symbols"]):
        if i in fragment:
            if symbol == "H":
                hydrogens.append(i)
            else:
                atom_order.append(i)
    atom_order.extend(hydrogens)
    return atom_order


def read_rdkit_mol_from_sdf(sdf_path: str, remove_hs: bool = False, sanitize: bool = True) -> Chem.Mol:
    """
    Docstring for read_rdkit_mol_from_sdf
    
    :param sdf_path: Description
    :type sdf_path: str
    :param remove_hs: Description
    :type remove_hs: bool
    :param sanitize: Description
    :type sanitize: bool
    :return: Description
    :rtype: Mol
    """
    rdmol = Chem.MolFromMolFile(
        sdf_path,
        removeHs=remove_hs,
        sanitize=sanitize,
        strictParsing=True
    )
    if rdmol is None:
        raise ZMatError(f"RDKit failed to read SDF: {sdf_path}")

    return rdmol


def read_rdkit_mols_from_sdf(
    sdf_path: str,
    remove_hs: bool = False,
    sanitize: bool = True,
) -> List[Chem.Mol]:
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=remove_hs, sanitize=sanitize)
    mols = [mol for mol in supplier if mol is not None]
    if not mols:
        raise ZMatError(f"RDKit failed to read SDF: {sdf_path}")
    return mols


def select_mol_from_sdf(
    sdf_path: str,
    prop_key: Optional[str] = None,
    prop_value: Optional[str] = None,
    index: int = 0,
    remove_hs: bool = False,
    sanitize: bool = True,
) -> Chem.Mol:
    mols = read_rdkit_mols_from_sdf(sdf_path, remove_hs=remove_hs, sanitize=sanitize)
    if prop_key is not None and prop_value is not None:
        for mol in mols:
            if mol.HasProp(prop_key) and mol.GetProp(prop_key) == prop_value:
                return mol
        raise ZMatError(
            f"Could not find molecule with {prop_key}={prop_value} in {sdf_path}"
        )
    if index < 0 or index >= len(mols):
        raise ZMatError(f"SDF has {len(mols)} molecules, index {index} is out of range")
    return mols[index]


def rdkit_mol_to_xyz_dict(rdmol: Chem.Mol, conf_id: int = -1) -> Dict[str, tuple]:
    """
    Docstring for rdkit_mol_to_xyz_dict
    
    :param rdmol: Description
    :type rdmol: Chem.Mol
    :param conf_id: Description
    :type conf_id: int
    :return: Description
    :rtype: Dict[str, tuple]
    """
    if rdmol.GetNumConformers() == 0:
        raise ZMatError("RDKit has no conformers (no 3D coordinates).")
    
    conf = rdmol.GetConformer(conf_id)
    symbols = []
    coords = []
    for i, atom in enumerate(rdmol.GetAtoms()):
        p = conf.GetAtomPosition(i)
        symbols.append(atom.GetSymbol())
        coords.append((float(p.x), float(p.y), float(p.z)))
    
    return {"symbols": tuple(symbols), "coords": tuple(coords)}



def validate_constraints_against_natoms(constraints: Dict[str, List[Tuple[int, ...]]], n_atoms: int) -> None:
    for constraint_list in constraints.values():
        for tpl in constraint_list:
            for idx in tpl:
                if idx < 0 or idx >= n_atoms:
                    raise ZMatError(
                        f"Constraint atom index {idx} is invalid for n_atoms{n_atoms}."
                        f"Constraints were:\n{constraints}"
                    )


def _add_nth_atom_to_zmat(zmat: Dict[str, Union[dict, tuple]],
                          xyz: Dict[str, tuple],
                          connectivity: Optional[Dict[int, Iterable[int]]],
                          n: int,
                          atom_index: int,
                          constraints: Dict[str, List[Tuple[int]]],
                          fragments: List[List[int]]) -> Tuple[Dict[str, tuple], Dict[str, tuple], List[int]]:
    """
    Add the n-th atom to the zmat (n >= 0).
    Also considers the special cases where ``n`` is the first, second, or third atom to be added to the zmat.
    Adds a dummy atom if an angle (not a dihedral angle) is 180 (or 0) degrees.
    
    :param zmat: Description
    :type zmat: Dict[str, Union[dict, tuple]]
    :param xyz: Description
    :type xyz: Dict[str, tuple]
    :param connectivity: Description
    :type connectivity: Optional[Dict[int, Iterable[int]]]
    :param n: Description
    :type n: int
    :param atom_index: Description
    :type atom_index: int
    :param constraints: Description
    :type constraints: Dict[str, List[Tuple[int]]]
    :param fragments: Description
    :type fragments: List[List[int]]
    :return: Description
    :rtype: Tuple[Dict[str, tuple], Dict[str, tuple], List[int]]
    """
    num_init_atoms = len(xyz["symbols"])
    coords = xyz["coords"]
    skipped_atoms = list()
    specific_last_d_atom = None
    r_constraint, r_constraint_type = check_atom_r_constraints(atom_index=atom_index, constraints=constraints)
    a_constraint, a_constraint_type = check_atom_a_constraints(atom_index=atom_index, constraints=constraints)
    d_constraint, d_constraint_type = check_atom_d_constraints(atom_index=atom_index, constraints=constraints)
    if (
        sum([constraint is not None for constraint in [r_constraint, a_constraint, d_constraint]]) > 1
        and not (
            r_constraint is not None
            and a_constraint is not None
            and r_constraint == tuple(a_constraint[:2])
        )
        and not (
            a_constraint is not None
            and d_constraint is not None
            and a_constraint == tuple(d_constraint[:3])
        )
    ):
        raise ZMatError(
            f"A single atom cannot be constrained by more than one constraint type, got:\n"
            f"R {r_constraint_type}: {r_constraint}\n"
            f"A {a_constraint_type}: {a_constraint}\n"
            f"D {d_constraint_type}: {d_constraint}"
        )
    r_constraint_passed, a_constraint_passed, d_constraint_passed = \
        [constraint is None or all([entry in list(zmat['map'].values()) for entry in constraint[1:]])
         for constraint in [r_constraint, a_constraint, d_constraint]]
    
    
    if all([passed for passed in [r_constraint_passed, a_constraint_passed, d_constraint_passed]]):
        # Add the n-th atom to the zmat
        
        # if an '_atom' was specified only consider this atom if n is the last atom to consider
        if (r_constraint_type == 'R_atom' or a_constraint_type == 'A_atom' or d_constraint_type == 'D_atom') \
            and n != num_init_atoms - 1:
                skipped_atoms.append(atom_index)
                return zmat, xyz, skipped_atoms

        r_atoms = determine_r_atoms(zmat, xyz, connectivity, n, atom_index, r_constraint, a_constraint, d_constraint,
                                    trivial_assignment=any('_atom'  in constraint_key for constraint_key in constraints.keys()),
                                    fragments=fragments)
        if a_constraint is None and d_constraint is not None:
            a_constraint = d_constraint[:3]
        
        a_atoms = determine_a_atoms(
            zmat,
            coords,
            connectivity,
            r_atoms,
            n,
            atom_index,
            a_constraint,
            d_constraint,
            a_constraint_type,
            trivial_assignment=any('_atom' in constraint_key for constraint_key in constraints.keys()),
            fragments=fragments,
        )
        
        added_dummy = False
        if a_atoms is not None and all([not re.match(r'X\d', str(zmat['map'][atom])) for atom in a_atoms[1:]]):
            angle = calculate_param(coords=coords, atoms=[atom_index] + [zmat['map'][atom] for atom in a_atoms[1:]])
            if is_angle_linear(angle, tolerance=TOL_180):
                # The angle is too close to 180 (or 0) degrees, add a dummy atom.
                zmat, coords, n, r_atoms, a_atoms, specific_last_d_atom = \
                    add_dummy_atom(zmat, xyz, coords, connectivity, r_atoms, a_atoms, n, atom_index)
                added_dummy = True

        d_atoms = determine_d_atoms(
            zmat,
            xyz,
            coords,
            connectivity,
            a_atoms,
            n,
            atom_index,
            d_constraint,
            d_constraint_type, specific_atom=specific_last_d_atom,
            fragments=fragments,
        )

        # Update the zmat.
        zmat = update_zmat_with_new_atom(zmat, xyz, coords, n, atom_index, r_atoms, a_atoms, d_atoms, added_dummy)

    else:
        # Some constraints did not "pass": some atoms were not added to the zmat yet; skip this atom until they are.
        skipped_atoms.append(atom_index)

    xyz['coords'] = coords  # Update xyz with the updated coords.
    return zmat, xyz, skipped_atoms

def xyz_to_zmat_rdkit(
    xyz: Dict[str, tuple],
    rdmol: Optional[Chem.Mol] = None,
    constraints: Optional[Dict[str, List[Tuple[int, ...]]]] = None,
    consolidate: bool = True,
    consolidation_tols: Optional[Dict[str, float]] = None,
    fragments: Optional[List[List[int]]] = None,
    atom_order: Optional[List[int]] = None,
    use_rmg_atom_order: bool = False,
) -> Dict[str, tuple]:
    """
    Docstring for xyz_to_zmat_rdkit
    
    :param xyz: Description
    :type xyz: Dict[str, tuple]
    :param rdmol: Description
    :type rdmol: Optional[Chem.Mol]
    :param constraints: Description
    :type constraints: Optional[Dict[str, List[Tuple[int, ...]]]]
    :param consolidate: Description
    :type consolidate: bool
    :param consolidation_tols: Description
    :type consolidation_tols: Optional[Dict[str, float]]
    :param fragments: Description
    :type fragments: Optional[List[List[int]]]
    :return: Description
    :rtype: Dict[str, tuple]
    """
    constraints = constraints or {}
    xyz = xyz.copy()
    
    n_atoms = len(xyz["symbols"])
    validate_constraints_against_natoms(constraints=constraints, n_atoms=n_atoms)
    
    if any("group" in k for k in constraints.keys()) and rdmol is None:
        raise ZMatError(
            "Cannot generate constrained zmat with *_group constaints without rdmol/connectivity"
            f"Got rdmol=None and constaints=\n{constraints}"
        )
        
    if rdmol is not None and use_rmg_atom_order:
        connectivity = get_rmg_connectivity_from_rdkit(rdmol) or get_connectivity_from_rdkit(rdmol=rdmol)
    else:
        connectivity = get_connectivity_from_rdkit(rdmol=rdmol) if rdmol is not None else None
    
    if fragments is None:
        if connectivity is not None:
            fragments = infer_fragments_from_connectivity(connectivity, n_atoms)
        else:
            fragments = [list(range(n_atoms))]
    
    
    zmat: Dict[str, Any] = {"symbols": [], "coords": [], "vars": {}, "map": {}}
    
    if atom_order is None:
        if use_rmg_atom_order and rdmol is not None:
            rmg_order = get_rmg_atom_order_from_rdkit(
                rdmol,
                constraints_dict=constraints,
                fragments=fragments,
                xyz=xyz,
            )
            if rmg_order is not None:
                atom_order = rmg_order
        if atom_order is None:
            atom_order = get_atom_order(
                xyz=xyz,
                rdmol=rdmol,
                fragments=fragments,
                constraints_dict=constraints,
            )
    elif sorted(atom_order) != list(range(n_atoms)):
        raise ZMatError(f"atom_order must contain all atom indices 0..{n_atoms - 1}, got {atom_order}")
    
    skipped_atoms: List[int] = []
    for atom_index in atom_order:
        zmat, xyz, skipped = _add_nth_atom_to_zmat(
            zmat=zmat,
            xyz=xyz,
            connectivity=connectivity,
            n=len(zmat["symbols"]),
            atom_index=atom_index,
            constraints=constraints,
            fragments=fragments
        )
        skipped_atoms.extend(skipped)
    while skipped_atoms:
        before = len(skipped_atoms)
        to_pop: List[int] = []
        for i, atom_index in enumerate(skipped_atoms):
            zmat, xyz, skipped = _add_nth_atom_to_zmat(
                zmat=zmat,
                xyz=xyz,
                connectivity=connectivity,
                n=len(zmat["symbols"]),
                atom_index=atom_index,
                constraints=constraints,
                fragments=fragments,
            )
            if not skipped:
                to_pop.append(i)

        for i in reversed(to_pop):
            skipped_atoms.pop(i)

        if before == len(skipped_atoms):
            raise ZMatError(
                "Could not generate zmat; skipped atoms could not be assigned. "
                f"Partial zmat:\n{zmat}\n\nskipped atoms:\n{skipped_atoms}\nconstraints:\n{constraints}"
            )

    if consolidate and not constraints:
        try:
            zmat = consolidate_zmat(zmat, None, consolidation_tols)  # <-- adjust consolidate_zmat to not require RMG
        except Exception:
            # keep behavior similar to your original
            pass

    zmat["symbols"] = tuple(zmat["symbols"])
    zmat["coords"] = tuple(zmat["coords"])
    return zmat
