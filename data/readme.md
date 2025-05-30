# About the Waypoints Data

`perching_high.csv`: Starts at a height higher than the hangar's height.  
`perching_high.csv`: Starts at the same height as the hangar's height.  
`perching_low.csv`: Starts at a height lower than the hangar's height.
 
 ## Generate command
```
./build/genTrajectory -i ./data/waypoints/perching_high.csv --v_max 1.0 --a_max 1.0 -o ./data/trajectories/perching_high_traj.csv

./build/genTrajectory -i ./data/waypoints/perching_mid1.csv --v_max 1.0 --a_max 1.0 -o ./data/trajectories/perching_mid_traj1.csv

./build/genTrajectory -i ./data/waypoints/perching_low.csv --v_max 1.0 --a_max 1.0 -o ./data/trajectories/perching_low_traj.csv

```

## Plot command

```

python3 ./scripts/plot_trajectory_lxl_3d.py ./data/trajectorys/perching_high_traj.csv --waypoints ./data/waypoints/perching_high.csv --real_data ./data/real_data/20250512_14_25_20.csv 
```