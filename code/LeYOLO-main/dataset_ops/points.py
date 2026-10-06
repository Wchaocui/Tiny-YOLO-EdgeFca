import numpy as np

# 打印当前使用的11个关键点坐标（单位：米）

def print_3d_keypoints(points_3d):
    """打印3D关键点坐标并分析分布"""
    print("3D关键点坐标（单位：米）:")
    for i, (x, y, z) in enumerate(points_3d):
        print(f"点{i + 1}: ({x:.4f}, {y:.4f}, {z:.4f})")

    # 分析关键点分布范围
    x_min, x_max = np.min(points_3d[:, 0]), np.max(points_3d[:, 0])
    y_min, y_max = np.min(points_3d[:, 1]), np.max(points_3d[:, 1])
    z_min, z_max = np.min(points_3d[:, 2]), np.max(points_3d[:, 2])

    print(f"\nX范围: {x_min:.4f} ~ {x_max:.4f} 米")
    print(f"Y范围: {y_min:.4f} ~ {y_max:.4f} 米")
    print(f"Z范围: {z_min:.4f} ~ {z_max:.4f} 米")

# 11个3D关键点（文档段落1-25）
points_3d = np.array([
    [-0.37, -0.37, 0.37, 0.37, -0.37, -0.37, 0.37, 0.37, -0.5427, 0.5427, 0.305],
    [-0.385, 0.385, 0.385, -0.385, -0.264, 0.304, 0.304, -0.264, 0.4877, 0.4877, -0.579],
    [0.3215, 0.3215, 0.3215, 0.3215, 0, 0, 0, 0, 0.2535, 0.2591, 0.2515]
]).T  # (11, 3)
# 使用当前定义的关键点
# print_3d_keypoints(points_3d)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_3d_keypoints(points_3d):
    """可视化3D关键点分布"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制关键点
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
               c='r', marker='o', s=50)

    # 绘制航天器轮廓（简化连接）
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底部矩形
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶部矩形
        [0, 4], [1, 5], [2, 6], [3, 7]  # 连接底部和顶部
    ]

    for edge in edges:
        ax.plot([points_3d[edge[0], 0], points_3d[edge[1], 0]],
                [points_3d[edge[0], 1], points_3d[edge[1], 1]],
                [points_3d[edge[0], 2], points_3d[edge[1], 2]], 'b-')

    # 设置坐标轴标签
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    # 设置视角
    ax.view_init(elev=30, azim=45)
    plt.title('航天器3D关键点分布')
    plt.show()


# 可视化关键点
visualize_3d_keypoints(points_3d)

