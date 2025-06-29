import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh

def refine_triangle_mesh_with_fealpy():
    # 1. 初始网格：单位直角三角形
    node = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    cell = np.array([[0, 1, 2]], dtype=np.int_)
    mesh = TriangleMesh(node, cell)

    # 2. 计算重心（作为新点）
    centroid = np.mean(node, axis=0)

    # 3. 边界吸附检查（阈值 epsilon = 0.05）
    epsilon = 0.05
    for i in range(3): #对三条边进行遍历检查，重心是否靠近边。
        p0 = node[i]
        p1 = node[(i + 1) % 3]
        edge_vec = p1 - p0 #边的方向向量
        edge_length = np.linalg.norm(edge_vec) #方向向量的范数
        t = np.dot(centroid - p0, edge_vec) / (edge_length ** 2) #计算重心在边上的投影参数t
        t_clipped = np.clip(t, 0.0, 1.0) #确保投影点在边上，而不是延长线上。
        projection = p0 + t_clipped * edge_vec #计算投影点的坐标projection
        dist = np.linalg.norm(centroid - projection)#计算重心到每一条边的投影距离dist

        #采用epsilon = 0.05来控制，若重心非常靠近边界，则将重心移动到该边的投影点projection
        if dist < epsilon:
            centroid = projection  # 吸附到边界
            break

    # 4. 添加新点并重新三角剖分
    new_node = np.vstack([node, centroid])
    new_cell = np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int_)
    refined_mesh = TriangleMesh(new_node, new_cell)

    # 5. 绘制网格
    fig = plt.figure(figsize=(8, 4))
    axes1 = fig.add_subplot(121)
    mesh.add_plot(axes1)
    axes1.set_title("Initial Mesh")
    axes2 = fig.add_subplot(122)
    refined_mesh.add_plot(axes2)
    axes2.set_title("Refined Mesh")
    plt.show()

    return refined_mesh

# 运行加密算法
refined_mesh = refine_triangle_mesh_with_fealpy()