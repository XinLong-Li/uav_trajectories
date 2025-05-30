#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

import uav_trajectory
import pandas as pd
from scipy.signal import savgol_filter

def sliding_window_linear_fit(t, x, window_size=21):
    """对每个点用滑动窗口线性拟合，返回速度数组"""
    half = window_size // 2
    v = np.zeros_like(x)
    for i in range(len(x)):
        i1 = max(0, i - half)
        i2 = min(len(x), i + half + 1)
        t_win = t[i1:i2]
        x_win = x[i1:i2]
        # 线性拟合: x = a*t + b
        A = np.vstack([t_win, np.ones_like(t_win)]).T
        a, _ = np.linalg.lstsq(A, x_win, rcond=None)[0]
        v[i] = a
    return v


def load_filtered_data(file_path):
    # 读取文件
    df = pd.read_csv(file_path, header=None, skiprows=1)

    # 过滤出时间值大于最后一行时间的行
    filtered_data = df.iloc[:-1]  # 跳过最后一行

    # 获取最后一行的时间值
    last_time = df.iloc[-1, 0]
    filtered_data = filtered_data[filtered_data[0] > last_time]

    # x_start = -0.877
    # filtered_data = filtered_data[filtered_data[1] > x_start]

    return filtered_data




if __name__ == "__main__":
        # 直接在这里填写参数
    trajectory_path = "./data/trajectories/final_high_traj.csv"
    waypoints_path = "./data/waypoints/final_high.csv"
    real_data_path = "./data/real_data/20250525_21_24_45.csv"
    stretchtime = 1.0  # 或 None
    # 加载轨迹
    traj = uav_trajectory.Trajectory()
    traj.loadcsv(trajectory_path)

    # 加载路径点
    waypoints = np.loadtxt(waypoints_path, delimiter=",")

    real_data = load_filtered_data(real_data_path)
    real_data_np = real_data.to_numpy()  # 转为 numpy 数组

    
    # 如果指定了时间拉伸因子，应用它
    if stretchtime:
        traj.stretchtime(stretchtime)

    # 评估轨迹
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

    # 计算速度、加速度和角速度的模
    velocity = np.linalg.norm(evals[:, 3:6], axis=1)
    acceleration = np.linalg.norm(evals[:, 6:9], axis=1)
    omega = np.linalg.norm(evals[:, 9:12], axis=1)

    # 打印统计信息
    print("max speed (m/s): ", np.max(velocity))
    print("max acceleration (m/s^2): ", np.max(acceleration))
    print("max omega (rad/s): ", np.max(omega))
    print("max roll (deg): ", np.max(np.degrees(evals[:, 13])))
    print("max pitch (deg): ", np.max(np.degrees(evals[:, 14])))


    # 创建 3D 轨迹图
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection="3d")  # 创建 3D 图
    ax_3d.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c="red", marker="o", s=50, label="Waypoints")  # 绘制路径点
    print('real-data')
    ax_3d.plot(evals[:, 0], evals[:, 1], evals[:, 2], label="ref")  # 绘制轨迹
   
    ax_3d.plot(real_data_np[:, 1], real_data_np[:, 2], real_data_np[:, 3], label="real")
    # ax_3d.plot(real_data[:, 1], real_data[:, 2], real_data[:, 3], label="real")  # 绘制轨迹
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

    # ...existing code...

    # # 画位置
    # fig1, axs1 = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    # axs1[0].plot(ts, evals[:, 0], label='x')
    # axs1[0].set_ylabel('X [m]')
    # axs1[1].plot(ts, evals[:, 1], label='y')
    # axs1[1].set_ylabel('Y [m]')
    # axs1[2].plot(ts, evals[:, 2], label='z')
    # axs1[2].set_ylabel('Z [m]')
    # axs1[2].set_xlabel('Time [s]')
    # axs1[0].set_title('Position')
    # for ax in axs1:
    #     ax.legend()
    #     ax.grid()

    # # 画速度
    # fig2, axs2 = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    # axs2[0].plot(ts, evals[:, 3], label='vx')
    # axs2[0].set_ylabel('VX [m/s]')
    # axs2[1].plot(ts, evals[:, 4], label='vy')
    # axs2[1].set_ylabel('VY [m/s]')
    # axs2[2].plot(ts, evals[:, 5], label='vz')
    # axs2[2].set_ylabel('VZ [m/s]')
    # axs2[2].set_xlabel('Time [s]')
    # axs2[0].set_title('Velocity')
    # for ax in axs2:
    #     ax.legend()
    #     ax.grid()

    # # 画加速度
    # fig3, axs3 = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    # axs3[0].plot(ts, evals[:, 6], label='ax')
    # axs3[0].set_ylabel('AX [m/s²]')
    # axs3[1].plot(ts, evals[:, 7], label='ay')
    # axs3[1].set_ylabel('AY [m/s²]')
    # axs3[2].plot(ts, evals[:, 8], label='az')
    # axs3[2].set_ylabel('AZ [m/s²]')
    # axs3[2].set_xlabel('Time [s]')
    # axs3[0].set_title('Acceleration')
    # for ax in axs3:
    #     ax.legend()
    #     ax.grid()

# ...existing code...


    # real_data_np: [时间, x, y, z, ...]
    t_real = real_data_np[:, 0]
    t_real = t_real - t_real[0]
    x_real = real_data_np[:, 1]
    y_real = real_data_np[:, 2]
    z_real = real_data_np[:, 3]

    
    # 先用滑动窗口线性拟合求速度
    vx_real = sliding_window_linear_fit(t_real, x_real, window_size=21)
    vy_real = sliding_window_linear_fit(t_real, y_real, window_size=21)
    vz_real = sliding_window_linear_fit(t_real, z_real, window_size=21)
    
    # 再用滑动窗口线性拟合求加速度
    ax_real = sliding_window_linear_fit(t_real, vx_real, window_size=21)
    ay_real = sliding_window_linear_fit(t_real, vy_real, window_size=21)
    az_real = sliding_window_linear_fit(t_real, vz_real, window_size=21)

    # # 先对位置做一次平滑
    # x_real_smooth = savgol_filter(x_real, window_length=101, polyorder=3)
    # y_real_smooth = savgol_filter(y_real, window_length=101, polyorder=3)
    # z_real_smooth = savgol_filter(z_real, window_length=101, polyorder=3)
    
    # # 再用 Savitzky-Golay 求导
    # vx_real = savgol_filter(x_real_smooth, window_length=101, polyorder=3, deriv=1, delta=np.mean(np.diff(t_real)))
    # vy_real = savgol_filter(y_real_smooth, window_length=101, polyorder=3, deriv=1, delta=np.mean(np.diff(t_real)))
    # vz_real = savgol_filter(z_real_smooth, window_length=101, polyorder=3, deriv=1, delta=np.mean(np.diff(t_real)))

    
    # 计算加速度（2阶导）
    # ax_real = savgol_filter(x_real_smooth, window_length=101, polyorder=3, deriv=2, delta=np.mean(np.diff(t_real)))
    # ay_real = savgol_filter(y_rx_real_smootheal, window_length=101, polyorder=3, deriv=2, delta=np.mean(np.diff(t_real)))
    # az_real = savgol_filter(x_real_smooth, window_length=101, polyorder=3, deriv=2, delta=np.mean(np.diff(t_real)))



    # x_real_smooth = savgol_filter(x_real, 101, 3)
    # y_real_smooth = savgol_filter(y_real, 101, 3)
    # z_real_smooth = savgol_filter(z_real, 101, 3)

    # # 再微分
    # vx_real = np.gradient(x_real_smooth, t_real)
    # vy_real = np.gradient(y_real_smooth, t_real)
    # vz_real = np.gradient(z_real_smooth, t_real)

    # # 计算速度
    # vx_real = np.gradient(x_real, t_real)
    # vy_real = np.gradient(y_real, t_real)
    # vz_real = np.gradient(z_real, t_real)

    # # 计算加速度
    # ax_real = np.gradient(vx_real, t_real)
    # ay_real = np.gradient(vy_real, t_real)
    # az_real = np.gradient(vz_real, t_real)

    # 画位置
    fig1, axs1 = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axs1[0].plot(ts, evals[:, 0], label='x_ref')
    axs1[0].plot(t_real, x_real, label='x_real', linestyle='--')
    axs1[0].set_ylabel('X [m]')
    axs1[1].plot(ts, evals[:, 1], label='y_ref')
    axs1[1].plot(t_real, y_real, label='y_real', linestyle='--')
    axs1[1].set_ylabel('Y [m]')
    axs1[2].plot(ts, evals[:, 2], label='z_ref')
    axs1[2].plot(t_real, z_real, label='z_real', linestyle='--')
    axs1[2].set_ylabel('Z [m]')
    axs1[2].set_xlabel('Time [s]')
    axs1[0].set_title('Position')
    for ax in axs1:
        ax.legend()
        ax.grid()
    # # 设置统一y轴范围
    # pos_min = min(np.min(evals[:, 0:3]), np.min(x_real), np.min(y_real), np.min(z_real))
    # pos_max = max(np.max(evals[:, 0:3]), np.max(x_real), np.max(y_real), np.max(z_real))
    # for ax in axs1:
    #     ax.set_ylim(pos_min, pos_max)

    # 画速度
    fig2, axs2 = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axs2[0].plot(ts, evals[:, 3], label='vx_ref')
    axs2[0].plot(t_real, vx_real, label='vx_real', linestyle='--')
    axs2[0].set_ylabel('VX [m/s]')
    axs2[1].plot(ts, evals[:, 4], label='vy_ref')
    axs2[1].plot(t_real, vy_real, label='vy_real', linestyle='--')
    axs2[1].set_ylabel('VY [m/s]')
    axs2[2].plot(ts, evals[:, 5], label='vz_ref')
    axs2[2].plot(t_real, vz_real, label='vz_real', linestyle='--')
    axs2[2].set_ylabel('VZ [m/s]')
    axs2[2].set_xlabel('Time [s]')
    axs2[0].set_title('Velocity')
    for ax in axs2:
        ax.legend()
        ax.grid()
    # 设置统一y轴范围
    vel_min = min(np.min(evals[:, 3:6]), np.min(vx_real), np.min(vy_real), np.min(vz_real))
    vel_max = max(np.max(evals[:, 3:6]), np.max(vx_real), np.max(vy_real), np.max(vz_real))
    for ax in axs2:
        ax.set_ylim(vel_min, vel_max)

    # 画加速度
    fig3, axs3 = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axs3[0].plot(ts, evals[:, 6], label='ax_ref')
    axs3[0].plot(t_real, ax_real, label='ax_real', linestyle='--')
    axs3[0].set_ylabel('AX [m/s²]')
    axs3[1].plot(ts, evals[:, 7], label='ay_ref')
    axs3[1].plot(t_real, ay_real, label='ay_real', linestyle='--')
    axs3[1].set_ylabel('AY [m/s²]')
    axs3[2].plot(ts, evals[:, 8], label='az_ref')
    axs3[2].plot(t_real, az_real, label='az_real', linestyle='--')
    axs3[2].set_ylabel('AZ [m/s²]')
    axs3[2].set_xlabel('Time [s]')
    axs3[0].set_title('Acceleration')
    for ax in axs3:
        ax.legend()
        ax.grid()
    # 设置统一y轴范围
    acc_min = min(np.min(evals[:, 6:9]), np.min(ax_real), np.min(ay_real), np.min(az_real))
    acc_max = max(np.max(evals[:, 6:9]), np.max(ax_real), np.max(ay_real), np.max(az_real))
    for ax in axs3:
        ax.set_ylim(acc_min, acc_max)

    plt.show()
# ...existing code...