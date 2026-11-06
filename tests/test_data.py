import numpy as np
import pytest

from wprl.data import load_structure, make_synthetic_structure, pair_structure_files, voxelize_points


def test_voxelize_places_points_and_keeps_max():
    xyz = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [100.0, 0.0, 0.0]])
    grid = voxelize_points(xyz, np.array([0.3, 0.9, 1.0]), grid_size=4, voxel_size=1.0)
    # (0.1 + 2) / 1 -> voxel index 2 on every axis; the far point is dropped.
    assert grid[2, 2, 2] == pytest.approx(0.9)
    assert grid.sum() == pytest.approx(0.9)


def test_pair_structure_files_matches_by_stem(tmp_path):
    for name in ["b.pdb", "a.pdb", "a_target_density.ccp4", "b_target_density.ccp4", "c.pdb"]:
        (tmp_path / name).write_text("")
    pairs = pair_structure_files(tmp_path)
    assert [(p.name, m.name) for p, m in pairs] == [
        ("a.pdb", "a_target_density.ccp4"),
        ("b.pdb", "b_target_density.ccp4"),
    ]


def test_synthetic_structure_is_well_formed():
    s = make_synthetic_structure(n_waters=10, grid_size=20, min_separation=2.5, seed=1)
    assert s.waters.shape == (10, 3) and s.density.shape == (20, 20, 20)
    assert s.density.min() >= 0 and s.density.max() == pytest.approx(1.0)
    d = np.linalg.norm(s.waters[:, None] - s.waters[None], axis=-1) + np.eye(10) * 99
    assert d.min() >= 2.5 - 1e-5
    assert np.allclose(s.waters.mean(axis=0), 0, atol=1e-5)
    assert s.n_out_of_bounds() == 0


def _write_test_files(tmp_path, waters, blob_centres=None, spacegroup="P1", cell=20.0, n=40, sigma=0.8):
    gemmi = pytest.importorskip("gemmi")
    # Map: periodic Gaussian blobs (by default at each water) in a cubic cell.
    axis = np.arange(n) / n * cell
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    grid_pts = np.stack([gx, gy, gz], -1)
    values = np.zeros((n, n, n), dtype=np.float32)
    for w in waters if blob_centres is None else blob_centres:
        d = grid_pts - w
        d -= cell * np.round(d / cell)  # minimum image
        values = np.maximum(values, np.exp(-(d**2).sum(-1) / (2 * sigma**2)))
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = gemmi.FloatGrid(n, n, n)
    ccp4.grid.unit_cell = gemmi.UnitCell(cell, cell, cell, 90, 90, 90)
    ccp4.grid.spacegroup = gemmi.SpaceGroup(spacegroup)
    np.array(ccp4.grid, copy=False)[:] = values
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(str(tmp_path / "x_target_density.ccp4"))

    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(cell, cell, cell, 90, 90, 90)
    st.spacegroup_hm = gemmi.SpaceGroup(spacegroup).hm
    model, chain = gemmi.Model("1"), gemmi.Chain("W")
    for i, w in enumerate(waters):
        res = gemmi.Residue()
        res.name, res.seqid = "HOH", gemmi.SeqId(i + 1, " ")
        atom = gemmi.Atom()
        atom.name, atom.element, atom.pos = "O", gemmi.Element("O"), gemmi.Position(*w)
        res.add_atom(atom)
        chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    st.write_pdb(str(tmp_path / "x.pdb"))
    return tmp_path / "x.pdb", tmp_path / "x_target_density.ccp4"


def test_load_structure_aligns_density_with_waters(tmp_path):
    # One water sits near the cell edge, so its blob is split across the
    # periodic boundary and must be reassembled from a neighbouring cell.
    waters = np.array([[8.0, 9.0, 10.0], [12.0, 11.0, 9.0], [10.0, 13.0, 19.6]])
    pdb, ccp4 = _write_test_files(tmp_path, waters)
    s = load_structure(pdb, ccp4, grid_size=32, voxel_size=0.5, density_threshold=0.3, pad=2.0)
    assert s.waters.shape == (3, 3)
    assert np.allclose(s.waters, waters - waters.mean(0), atol=1e-4)
    idx = np.floor((s.waters + s.half_extent) / s.voxel_size).astype(int)
    peak_values = s.density[idx[:, 0], idx[:, 1], idx[:, 2]]
    assert np.all(peak_values > 0.8), peak_values


def test_load_structure_applies_symmetry_operations(tmp_path):
    # In P2_1 the map only holds the symmetry mate (-x, y + 1/2, -z) of the
    # water; the loader must apply the operation to put density on the water.
    cell = 20.0
    water = np.array([6.0, 7.0, 5.0])
    mate = np.array([-water[0], water[1] + cell / 2, -water[2]]) % cell
    pdb, ccp4 = _write_test_files(tmp_path, water[None], blob_centres=mate[None], spacegroup="P 1 21 1")
    s = load_structure(pdb, ccp4, grid_size=16, voxel_size=0.5, density_threshold=0.3, pad=2.0)
    centre = s.grid_size // 2
    assert s.density[centre, centre, centre] > 0.8
