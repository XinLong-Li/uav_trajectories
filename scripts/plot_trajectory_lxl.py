#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import argparse

import uav_trajectory

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("trajectory", type=str, help="CSV file containing trajectory")
  parser.add_argument("--stretchtime", type=float, help="stretch time factor (smaller means faster)")
  args = parser.parse_args()

  traj = uav_trajectory.Trajectory()
  traj.loadcsv(args.trajectory)

  if args.stretchtime:
    traj.stretchtime(args.stretchtime)

  ts = np.arange(0, traj.duration, 0.01)
  evals = np.empty((len(ts), 15))
  for t, i in zip(ts, range(0, len(ts))):
    e = traj.eval(t)
    evals[i, 0:3]  = e.pos
    evals[i, 3:6]  = e.vel
    evals[i, 6:9]  = e.acc
    evals[i, 9:12] = e.omega
    evals[i, 12]   = e.yaw
    evals[i, 13]   = e.roll
    evals[i, 14]   = e.pitch

  velocity = np.linalg.norm(evals[:,3:6], axis=1)
  acceleration = np.linalg.norm(evals[:,6:9], axis=1)
  omega = np.linalg.norm(evals[:,9:12], axis=1)

  # print stats
  print("max speed (m/s): ", np.max(velocity))
  print("max acceleration (m/s^2): ", np.max(acceleration))
  print("max omega (rad/s): ", np.max(omega))
  print("max roll (deg): ", np.max(np.degrees(evals[:,13])))
  print("max pitch (deg): ", np.max(np.degrees(evals[:,14])))

  waypoints = np.loadtxt("/home/lxl/uav_trajectories/data/waypoints/perching_mid.csv", delimiter=",") 
 


  # Create a separate figure for the 3D trajectory plot
  fig_3d = plt.figure()
  ax_3d = fig_3d.add_subplot(111, projection='3d')  # Create a 3D plot
  ax_3d.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c='red', marker='o', s=50, label="Waypoints")  # Plot waypoints
  ax_3d.plot(evals[:,0], evals[:,1], evals[:,2], label="Trajectory")    # Plot the trajectory
  ax_3d.set_xlabel("X [m]")                        # Set X-axis label
  ax_3d.set_ylabel("Y [m]")                        # Set Y-axis label
  ax_3d.set_zlabel("Z [m]")                        # Set Z-axis label
  ax_3d.set_title("3D Trajectory")                # Set the title
  ax_3d.legend()
  
  # Set equal scaling for all axes
  x_limits = ax_3d.get_xlim()
  y_limits = ax_3d.get_ylim()
  z_limits = ax_3d.get_zlim()

  # Find the max range for all axes
  max_range = max(
      x_limits[1] - x_limits[0],
      y_limits[1] - y_limits[0],
      z_limits[1] - z_limits[0]
  ) / 2.0

  # Calculate the midpoints for each axis
  x_mid = (x_limits[0] + x_limits[1]) / 2.0
  y_mid = (y_limits[0] + y_limits[1]) / 2.0
  z_mid = (z_limits[0] + z_limits[1]) / 2.0

  # Set the limits for each axis to ensure equal scaling
  ax_3d.set_xlim([x_mid - max_range, x_mid + max_range])
  ax_3d.set_ylim([y_mid - max_range, y_mid + max_range])
  ax_3d.set_zlim([z_mid - max_range, z_mid + max_range])

  plt.show()

