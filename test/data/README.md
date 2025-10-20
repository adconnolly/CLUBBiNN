# Data Files

- `BOMEX_pruned_thinned.nc` - The original BOMEX dataset used in development
  but pruned to relevant variables used in this project and thinned in
  samples to reduce size whilst maintaining sample space coverage.
  Generated using `thin_bomex.py`.
- `bomex_mixing_length_calculation_samples.nc` - Sample invocations of the
  `compute_mixing_length` in CLUBB (obtained by logging inputs and outputs
  during standalone calculation). Used to check that the Python reimplementation
  of the same function is equivalent.
- `reference_CLUBB_zt_grid.csv`: Values of the thermodynamic CLUBB grid build 
  by coarsening BOMEX LES results. Used in a regression test to check if grid 
  construction matches what was done initially 
- `reference_CLUBB_zm_grid.csv`: Values of the CLUBB momentum grid that match 
  the `zt` in the file above.
