# Data layout

Structural data is not committed to the repository. Put each structure and
its target density map side by side, named so the map shares the PDB file's
stem:

```
data/
  train/
    1abc.pdb
    1abc_target_density.ccp4
    2xyz.pdb
    2xyz_target_density.ccp4
  test/
    3pqr.pdb
    3pqr_target_density.ccp4
```

Then point a config at the folders:

```yaml
data:
  synthetic: null
  train_dir: data/train
  test_dir: data/test
  grid_size: 48
  voxel_size: 1.0
  density_threshold: 0.5
```

Keep `train/` and `test/` disjoint; evaluation only runs on `test_dir`.
Loading needs the optional `gemmi` dependency: `pip install -e ".[structures]"`.
