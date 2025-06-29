#没有调用fealpy.backend_manager，因为个人电脑上所安装的fealpy包有问题，目前还未解决。
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh

def build_mesh():
    # 初始网格（一个正三角形）
    node = np.array([[0.0, 0.0], [2.0, 0.0], [1, np.sqrt(3)]], dtype=np.float64)
    cell = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = TriangleMesh(node, cell)
    
    # 全局加密 2 次
    mesh.uniform_refine(2)
    
    # 对最后一个单元的节点进行扰动
    cell = mesh.entity('cell')
    node = mesh.entity('node')
    node[cell[-1, 0]] += [-0.15, 0.05]
    node[cell[-1, 1]] += [-0.1, 0.15]
    node[cell[-1, 2]] += [0, -0.15]
    
    # 再全局加密 3 次
    mesh.uniform_refine(3)
    return mesh

def compute_angles(mesh):
    """计算所有三角形的三个内角（角度制）"""
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    angles = []
    
    for tri in cell:
        A, B, C = node[tri[0]], node[tri[1]], node[tri[2]]
        
        # 计算向量
        AB = B - A
        AC = C - A
        BA = A - B
        BC = C - B
        CA = A - C
        CB = B - C
        
        # 计算角度（使用点积公式）
        angle_A = np.arccos(np.dot(AB, AC) / (np.linalg.norm(AB) * np.linalg.norm(AC)))
        angle_B = np.arccos(np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC)))
        angle_C = np.arccos(np.dot(CA, CB) / (np.linalg.norm(CA) * np.linalg.norm(CB)))
        
        angles.extend([np.degrees(angle_A), np.degrees(angle_B), np.degrees(angle_C)])
    
    return np.array(angles)

# 可视化 
def plot_quality(angles, mesh):
        """绘制三角形角度分布的直方图和网格"""  
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # 角度分布直方图    
        axs[0].hist(angles, bins=30, color='skyblue', edgecolor='black')
        axs[0].set_xlabel('Angle (degrees)')
        axs[0].set_ylabel('Count')
        axs[0].set_title('Triangle Angle Distribution')
        
        # 网格可视化
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        for tri in cell:
            polygon = node[tri]
            axs[1].fill(*zip(*polygon), edgecolor='black', fill=False)
        axs[1].set_aspect('equal')
        axs[1].set_title('Mesh')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('y')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 生成网格
    mesh = build_mesh()
    
    # 计算角度分布
    angles = compute_angles(mesh)
    print("角度统计：")
    print(f"  最小值：{np.min(angles):.2f}°")
    print(f"  最大值：{np.max(angles):.2f}°")
    print(f"  平均值：{np.mean(angles):.2f}°")
    print(f"  中位数：{np.median(angles):.2f}°")

    plot_quality(angles, mesh)