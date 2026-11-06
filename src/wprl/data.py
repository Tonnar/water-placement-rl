"""Loading structures and density maps into a fixed-size voxel grid.

A :class:`Structure` holds the ground-truth water positions (centred on their
mean) and a cubic density grid on the same coordinate frame. Real data come
from a PDB file plus a CCP4 map (:func:`load_structure`); synthetic data with
Gaussian blobs at each water (:func:`make_synthetic_structure`) let the
environments, agents and tests run without any files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

WATER_RESIDUE_NAMES = ("HOH", "WAT", "H2O")


@dataclass
class Structure:
    """Water targets and density on a cubic grid centred at the origin.

    Attributes:
        waters: (N, 3) float32 target water-oxygen positions, in Angstrom.
        density: (G, G, G) float32 density, normalised to [0, 1].
        voxel_size: edge length of one voxel, in Angstrom.
        name: identifier, usually the PDB file stem.
    """

    waters: np.ndarray
    density: np.ndarray
    voxel_size: float
    name: str = ""

    @property
    def grid_size(self) -> int:
        return int(self.density.shape[0])

    @property
    def half_extent(self) -> float:
        return self.grid_size * self.voxel_size / 2.0

    @property
    def n_waters(self) -> int:
        return int(self.waters.shape[0])

    def n_out_of_bounds(self) -> int:
        """Number of target waters that fall outside the density grid."""
        return int(np.any(np.abs(self.waters) > self.half_extent, axis=1).sum())


def voxelize_points(
    xyz: np.ndarray, values: np.ndarray, grid_size: int, voxel_size: float
) -> np.ndarray:
    """Scatter point values into a cubic grid centred at the origin.

    Each voxel keeps the maximum value of the points that fall inside it;
    points outside the grid are dropped.
    """
    grid = np.zeros((grid_size,) * 3, dtype=np.float32)
    if len(xyz) == 0:
        return grid
    half_extent = grid_size * voxel_size / 2.0
    idx = np.floor((np.asarray(xyz) + half_extent) / voxel_size).astype(np.int64)
    inside = np.all((idx >= 0) & (idx < grid_size), axis=1)
    idx, vals = idx[inside], np.asarray(values, dtype=np.float32)[inside]
    np.maximum.at(grid, (idx[:, 0], idx[:, 1], idx[:, 2]), vals)
    return grid


def _normalise(values: np.ndarray) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def load_structure(
    pdb_path: str | Path,
    ccp4_path: str | Path,
    grid_size: int = 48,
    voxel_size: float = 1.0,
    density_threshold: float = 0.5,
    pad: float = 1.0,
) -> Structure:
    """Load waters from a PDB file and their density from a CCP4 map.

    Map values above ``density_threshold`` are expanded by every space-group
    symmetry operation and by neighbouring unit cells, cropped to a box
    around the waters (plus ``pad``), centred on the waters' mean and
    voxelised.

    Requires the optional ``gemmi`` dependency.
    """
    import gemmi  # optional dependency, imported lazily

    structure = gemmi.read_structure(str(pdb_path))
    waters = np.array(
        [
            [atom.pos.x, atom.pos.y, atom.pos.z]
            for chain in structure[0]
            for residue in chain
            if residue.name in WATER_RESIDUE_NAMES
            for atom in residue
        ],
        dtype=np.float64,
    ).reshape(-1, 3)
    if len(waters) == 0:
        raise ValueError(f"{pdb_path}: no water molecules found")

    ccp4 = gemmi.read_ccp4_map(str(ccp4_path), setup=True)
    grid = ccp4.grid
    values = np.array(grid, copy=False)
    nu, nv, nw = values.shape

    # Fractional coordinates of the grid nodes above the threshold.
    # Grid node (i, j, k) sits at (i/nu, j/nv, k/nw).
    idx = np.argwhere(values > density_threshold)
    node_values = values[values > density_threshold].astype(np.float64)
    frac = idx / np.array([nu, nv, nw], dtype=np.float64)

    orth = np.array(grid.unit_cell.orth.mat.tolist())
    lo = waters.min(axis=0) - pad
    hi = waters.max(axis=0) + pad

    shifts = np.array(
        [[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)],
        dtype=np.float64,
    )
    kept_xyz, kept_vals = [], []
    den = float(gemmi.Op.DEN)
    for op in grid.spacegroup.operations():
        rot = np.array(op.rot, dtype=np.float64) / den
        tran = np.array(op.tran, dtype=np.float64) / den
        sym_frac = frac @ rot.T + tran
        for shift in shifts:
            xyz = (sym_frac + shift) @ orth.T
            in_box = np.all((xyz >= lo) & (xyz <= hi), axis=1)
            if in_box.any():
                kept_xyz.append(xyz[in_box])
                kept_vals.append(node_values[in_box])

    center = waters.mean(axis=0)
    if kept_xyz:
        xyz = np.concatenate(kept_xyz) - center
        vals = _normalise(np.concatenate(kept_vals))
    else:
        xyz, vals = np.zeros((0, 3)), np.zeros(0, dtype=np.float32)

    return Structure(
        waters=(waters - center).astype(np.float32),
        density=voxelize_points(xyz, vals, grid_size, voxel_size),
        voxel_size=voxel_size,
        name=Path(pdb_path).stem,
    )


def pair_structure_files(
    directory: str | Path, map_suffix: str = "_target_density.ccp4"
) -> list[tuple[Path, Path]]:
    """Pair ``X.pdb`` with ``X<map_suffix>`` in a directory, sorted by name.

    Pairing by file stem matters: two separate ``glob`` calls return files in
    arbitrary order, which can silently match a structure with the wrong map.
    """
    directory = Path(directory)
    pairs = []
    for pdb in sorted(directory.glob("*.pdb")):
        ccp4 = directory / f"{pdb.stem}{map_suffix}"
        if ccp4.exists():
            pairs.append((pdb, ccp4))
    return pairs


def load_directory(
    directory: str | Path, limit: int | None = None, **kwargs
) -> list[Structure]:
    """Load every PDB/CCP4 pair in ``directory`` (see :func:`load_structure`)."""
    pairs = pair_structure_files(directory)
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        raise FileNotFoundError(f"no PDB/CCP4 pairs found in {directory}")
    return [load_structure(pdb, ccp4, **kwargs) for pdb, ccp4 in pairs]


def make_synthetic_structure(
    n_waters: int = 16,
    grid_size: int = 24,
    voxel_size: float = 1.0,
    sigma: float = 1.0,
    min_separation: float = 2.6,
    seed: int | None = None,
    name: str = "synthetic",
) -> Structure:
    """Random, well-separated waters with a Gaussian density blob at each.

    Useful for tests and for checking that an agent can learn at all before
    spending GPU time on real maps.
    """
    rng = np.random.default_rng(seed)
    half = grid_size * voxel_size / 2.0
    margin = 3.0 * voxel_size
    waters: list[np.ndarray] = []
    attempts = 0
    while len(waters) < n_waters:
        attempts += 1
        if attempts > 100_000:
            raise ValueError("could not place waters; lower n_waters or min_separation")
        p = rng.uniform(-half + margin, half - margin, size=3)
        if all(np.linalg.norm(p - q) >= min_separation for q in waters):
            waters.append(p)
    w = np.array(waters, dtype=np.float32)
    w -= w.mean(axis=0)

    axis = (np.arange(grid_size) + 0.5) * voxel_size - half
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    centres = np.stack([gx, gy, gz], axis=-1)[..., None, :]  # (G,G,G,1,3)
    d2 = ((centres - w) ** 2).sum(-1)  # (G,G,G,N)
    density = np.exp(-d2 / (2 * sigma**2)).max(-1)
    return Structure(
        waters=w,
        density=_normalise(density),
        voxel_size=voxel_size,
        name=name,
    )
