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

def compute_radius_ratio(mesh):
    """计算所有三角形的半径比 μ = R/(2r)"""
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    radius_ratios = []
    
    for tri in cell:
        A, B, C = node[tri[0]], node[tri[1]], node[tri[2]]
        
        # 计算边长
        a = np.linalg.norm(B - C)
        b = np.linalg.norm(A - C)
        c = np.linalg.norm(A - B)
        
        # 计算半周长和面积
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        
        # 内接圆半径 r = area / s
        r = area / s
        
        # 外接圆半径 R = abc / (4 * area)
        R = (a * b * c) / (4 * area)
        
        # 半径比 μ = R / (2r)
        mu = R / (2 * r)
        radius_ratios.append(mu)
    
    return np.array(radius_ratios)

def plot_quality(angles, radius_ratios):
    """可视化角度分布和半径比分布"""
    plt.figure(figsize=(12, 5))
    
    # 角度分布
    plt.subplot(1, 2, 1)
    plt.hist(angles, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(60, color='red', linestyle='--', label='理想角度 (60°)')
    plt.xlabel('角度 (度)')
    plt.ylabel('频数')
    plt.title('三角形角度分布')
    plt.legend()
    
    # 半径比分布
    plt.subplot(1, 2, 2)
    plt.hist(radius_ratios, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(1, color='red', linestyle='--', label='理想半径比 (μ=1)')
    plt.xlabel('半径比 μ = R/(2r)')
    plt.ylabel('频数')
    plt.title('三角形半径比分布')
    plt.legend()
    
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
    
    # 计算半径比
    radius_ratios = compute_radius_ratio(mesh)
    print("\n半径比统计：")
    print(f"  最小值：{np.min(radius_ratios):.4f}")
    print(f"  最大值：{np.max(radius_ratios):.4f}")
    print(f"  平均值：{np.mean(radius_ratios):.4f}")
    print(f"  中位数：{np.median(radius_ratios):.4f}")
    
    # 可视化
    plot_quality(angles, radius_ratios)