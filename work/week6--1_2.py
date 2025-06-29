import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import QuadrangleMesh

def quadtree_refine_with_fealpy():
    # 1. 初始网格：单位正方形（4个顶点）
    node = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    cell = np.array([[0, 1, 2, 3]], dtype=np.int_)
    mesh = QuadrangleMesh(node, cell)

    # 2. 执行四叉树加密（手动实现）
    # 计算四条边的中点
    mid_01 = (node[0] + node[1]) / 2  # 下边中点
    mid_12 = (node[1] + node[2]) / 2  # 右边中点
    mid_23 = (node[2] + node[3]) / 2  # 上边中点
    mid_30 = (node[3] + node[0]) / 2  # 左边中点
    # 计算中心点
    center = np.mean(node, axis=0)

    # 新节点集合（原节点 + 中点 + 中心点）
    new_node = np.vstack([node, mid_01, mid_12, mid_23, mid_30, center])

    # 新单元连接关系（4个子四边形）
    new_cell = np.array([
        [0, 4, 8, 7],   # 左下子四边形
        [4, 1, 5, 8],    # 右下子四边形
        [8, 5, 2, 6],    # 右上子四边形
        [7, 8, 6, 3]     # 左上子四边形
    ], dtype=np.int_)

    refined_mesh = QuadrangleMesh(new_node, new_cell)

    # 3. 绘制网格
    fig = plt.figure(figsize=(8, 4))
    axes1 = fig.add_subplot(121)
    mesh.add_plot(axes1)
    axes1.set_title("Initial Mesh")
    axes2 = fig.add_subplot(122)
    refined_mesh.add_plot(axes2)
    axes2.set_title("Refined Mesh (Quadtree)")
    plt.show()

    return refined_mesh

# 运行四叉树加密算法
refined_mesh = quadtree_refine_with_fealpy()