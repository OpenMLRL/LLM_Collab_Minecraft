# 2d_painting dataset

Each task is a JSON file with:

- `task_id`
- `palette`: list of 10 allowed block ids
- `bbox`: fixed build area (local coords)
- `target_spec.grid_rows`: 16 strings, each 16 digits (0-9)

Index mapping: `idx = int(grid_rows[y][x])`, block id is `palette[idx]`.
