from pathlib import Path

from heuristics.sdf_to_zmat import (
    get_connectivity_from_rdkit,
    infer_fragments_from_connectivity,
    read_rdkit_mol_from_sdf,
    rdkit_mol_to_xyz_dict,
    xyz_to_zmat_rdkit,
)


def _sample_sdf_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "DATA" / "SDF" / "deduped_sdf" / "kfir_rxn_10218.sdf"


def test_xyz_to_zmat_rdkit_smoke():
    sdf_path = _sample_sdf_path()
    rdmol = read_rdkit_mol_from_sdf(str(sdf_path))
    xyz = rdkit_mol_to_xyz_dict(rdmol)
    zmat = xyz_to_zmat_rdkit(xyz=xyz, rdmol=rdmol)

    assert len(zmat["symbols"]) == len(zmat["coords"])
    assert len(zmat["map"]) == len(zmat["symbols"])
    assert all(len(entry) == 3 for entry in zmat["coords"])


def test_xyz_to_zmat_rdkit_r_atom_constraint_skips_then_resolves():
    sdf_path = _sample_sdf_path()
    rdmol = read_rdkit_mol_from_sdf(str(sdf_path))
    xyz = rdkit_mol_to_xyz_dict(rdmol)

    constraints = {"R_atom": [(0, 1)]}
    zmat = xyz_to_zmat_rdkit(xyz=xyz, rdmol=rdmol, constraints=constraints)

    assert 0 in zmat["map"].values()


def test_infer_fragments_single_component():
    sdf_path = _sample_sdf_path()
    rdmol = read_rdkit_mol_from_sdf(str(sdf_path))
    connectivity = get_connectivity_from_rdkit(rdmol)
    fragments = infer_fragments_from_connectivity(connectivity, rdmol.GetNumAtoms())

    assert len(fragments) == 1
    assert fragments[0] == list(range(rdmol.GetNumAtoms()))


def test_xyz_to_zmat_respects_atom_order():
    sdf_path = _sample_sdf_path()
    rdmol = read_rdkit_mol_from_sdf(str(sdf_path))
    xyz = rdkit_mol_to_xyz_dict(rdmol)
    n_atoms = rdmol.GetNumAtoms()

    atom_order = list(reversed(range(n_atoms)))
    zmat = xyz_to_zmat_rdkit(xyz=xyz, rdmol=rdmol, atom_order=atom_order)

    assert zmat["map"][0] == atom_order[0]
