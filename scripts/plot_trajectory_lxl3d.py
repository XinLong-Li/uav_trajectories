#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

import uav_trajectory
import pandas as pd


def load_filtered_data(file_path):
    """
    Load and filter the trajectory data from a CSV file.
    Args:
        file_path (str): Path to the CSV file containing trajectory data.
    Returns:
        pd.DataFrame: Filtered trajectory data excluding the last row and with x > -0.877.
    """
    df = pd.read_csv(file_path, header=None, skiprows=1)

    filtered_data = df.iloc[:-1]  # Skip the last row. It records the time when the trajectory starts to execute.
    # last_time = df.iloc[-1, 0]
    # filtered_data = filtered_data[filtered_data[0] > last_time]

    x_start = -0.877
    filtered_data = filtered_data[filtered_data[1] > x_start]

    return filtered_data


if __name__ == "__main__":

    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection="3d")


    # Plot waypoints
    waypoints_path = "./data/waypoints/final_middle.csv"
    waypoints = np.loadtxt(waypoints_path, delimiter=",")
    ax_3d.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c="red", marker="o", s=50, label="Waypoints")


    # Plot commanded trajectory
    traj = uav_trajectory.Trajectory()
    trajectory_path = "./data/trajectories/final_middle_traj.csv"
    traj.loadcsv(trajectory_path)
    ts = np.arange(0, traj.duration, 0.01)
    evals = np.empty((len(ts), 15))
    for t, i in zip(ts, range(0, len(ts))):
        e = traj.eval(t)
        evals[i, 0:3] = e.pos
        evals[i, 3:6] = e.vel
        evals[i, 6:9] = e.acc
        evals[i, 9:12] = e.omega
        evals[i, 12] = e.yaw
        evals[i, 13] = e.roll
        evals[i, 14] = e.pitch
    ax_3d.plot(evals[:, 0], evals[:, 1], evals[:, 2], label="ref")  # 绘制轨迹


    # Plot real trajectory   
    real_data_path = "./data/real_data/archive/20250525_23_04_28.csv"
    real_data = load_filtered_data(real_data_path)
    real_data_np = real_data.to_numpy()  # 转为 numpy 数组
    ax_3d.plot(real_data_np[:, 1], real_data_np[:, 2], real_data_np[:, 3], label="real")
    # ax_3d.plot(real_data[:, 1], real_data[:, 2], real_data[:, 3], label="real")  # 绘制轨迹

    # 绘制立方体方框
    x_min, x_max = 0.155-0.032, 0.377+0.032
    y_min, y_max = -0.122-0.032, 0.293+0.032
    z_min, z_max = 0.741, 0.834

    # 8个顶点
    corners = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max],
    ])

    # 12条棱，每条棱由两个顶点组成
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7],  # 侧面
    ]

    for edge in edges:
        ax_3d.plot(
            [corners[edge[0], 0], corners[edge[1], 0]],
            [corners[edge[0], 1], corners[edge[1], 1]],
            [corners[edge[0], 2], corners[edge[1], 2]],
            c="g", linewidth=2, label=None
        )

    ax_3d.set_xlabel("X [m]")  # 设置 X 轴标签
    ax_3d.set_ylabel("Y [m]")  # 设置 Y 轴标签
    ax_3d.set_zlabel("Z [m]")  # 设置 Z 轴标签
    ax_3d.set_title("3D Trajectory")  # 设置标题
    ax_3d.legend()

    # 设置所有轴的比例相同
    x_limits = ax_3d.get_xlim()
    y_limits = ax_3d.get_ylim()
    z_limits = ax_3d.get_zlim()

    max_range = max(
        x_limits[1] - x_limits[0],
        y_limits[1] - y_limits[0],
        z_limits[1] - z_limits[0],
    ) / 2.0

    x_mid = (x_limits[0] + x_limits[1]) / 2.0
    y_mid = (y_limits[0] + y_limits[1]) / 2.0
    z_mid = (z_limits[0] + z_limits[1]) / 2.0

    ax_3d.set_xlim([x_mid - max_range, x_mid + max_range])
    ax_3d.set_ylim([y_mid - max_range, y_mid + max_range])
    ax_3d.set_zlim([z_mid - max_range, z_mid + max_range])

    plt.tight_layout()
    plt.show()