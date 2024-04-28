# https://www.zivid.com/3d-point-cloud-examples/pcb-motherboard-3d-point-cloud-example
# https://sketchfab.com/zivid/collections/inspection-and-fine-details-dcb240397e5849fbacd73748e8129ace

import open3d as o3d

def visualize_ply(ply_file_path):
    # 加载 PLY 文件
    point_cloud = o3d.io.read_point_cloud(ply_file_path)

    # 打印点云信息，查看点云结构
    print(point_cloud)

    # 可视化点云
    o3d.visualization.draw_geometries([point_cloud])

# 指定你的 .ply 文件路径
file_path = r'C:\Users\19002\Desktop\cloud_lesson\3D-visual-defect-detection\lesson3\welding_scene.ply'
visualize_ply(file_path)
