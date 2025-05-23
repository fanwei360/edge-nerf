import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import numpy as np
import open3d as o3d
from models.rendering import *
from models.nerf import *
import matplotlib.pyplot as plt

from utils import load_ckpt


def plt_vis(pcd, save_path):
    points = pcd.points

    vis_points = (np.array(points) / 256 * 2.4) - 1.2
    vis_points = np.array([[pts[1], pts[0], pts[2]] for pts in vis_points])

    x = [k[0] for k in vis_points]
    y = [k[1] for k in vis_points]
    z = [k[2] for k in vis_points]

    fig = plt.figure(dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    # plt.title('point cloud')
    ax.view_init(azim=60, elev=60)

    ax.scatter(x, y, z, c='black', marker='.', s=2, linewidth=0, alpha=1, cmap='spectral')

    range_size = [-1, 1]
    ax.set_zlim3d(range_size[0], range_size[1])
    plt.axis([range_size[0], range_size[1], range_size[0], range_size[1]])
    plt.axis('off')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def filter_soft_edge(pcd_raw):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_raw)
    density = np.array([color[0] for color in pcd_raw.colors])
    sigma_threshold2 = np.mean(density)

    soft_edge_index = np.argwhere(density < sigma_threshold2)
    inds = list(np.argwhere(density >= sigma_threshold2))
    for each_index in soft_edge_index:
        soft_edge = pcd_raw.points[each_index]
        radius = 4
        [num_radius, idx_radius, _] = pcd_tree.search_radius_vector_3d(soft_edge, radius)
        near_densities = density[idx_radius]

        temp = []
        for each in near_densities:
            if each > sigma_threshold2:
                temp.append(1)
            else:
                temp.append(0)
        if sum(temp) > 4:
            inds.append(each_index)

    print("points after filtering soft edge:", len(inds))
    final_pcd = pcd.select_by_index(inds)
    return final_pcd


def get_sigma_from_nerf(ckpt_path, size=1.2, N=128):
    chunk = 1024 * 32
    # dataset = dataset_dict[dataset_name](**kwargs)

    embedding_xyz = Embedding(10)
    embedding_dir = Embedding(4)

    nerf_fine = NeRF(in_channels_xyz=63, in_channels_dir=27)
    load_ckpt(nerf_fine, ckpt_path, model_name='nerf_fine')
    nerf_fine.cuda().eval()
    # print(nerf_fine)


    xmin, xmax = -size, size  # left/right range
    ymin, ymax = -size, size  # forward/backward range
    zmin, zmax = -size, size  # up/down range

    x = np.linspace(xmin, xmax, N)
    y = -np.linspace(ymin, ymax, N)  # inverse y axis
    z = np.linspace(zmin, zmax, N)

    xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).cuda()
    dir_ = torch.zeros_like(xyz_).cuda()

    print('Predicting occupancy ...')
    with torch.no_grad():
        B = xyz_.shape[0]
        out_chunks = []
        for i in range(0, B, chunk):
            xyz_embedded = embedding_xyz(xyz_[i:i + chunk])  # (N, embed_xyz_channels)
            dir_embedded = embedding_dir(dir_[i:i + chunk])  # (N, embed_dir_channels)
            xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded], 1)
            out_chunks += [nerf_fine(xyzdir_embedded, 'edge')]
        rgbsigma = torch.cat(out_chunks, 0)

    gray = rgbsigma[:, 0].cpu().numpy()
    gray = np.maximum(gray, 0)
    gray = gray.reshape(N, N, N)
    sigma = rgbsigma[:, -2].cpu().numpy()
    sigma_ = rgbsigma[:, -1].cpu().numpy()
    sigma = (sigma_>0)*sigma
    sigma = np.maximum(sigma, 0)
    sigma = sigma.reshape(N, N, N)
    return sigma, gray

def mapping_coor(points, size, N):
    points = (points - N/2) * 2 * size / N
    return points


target_path = r"/home/wei/NeRF_src/edge-nerf/pointcloud/house01"
os.makedirs(target_path, exist_ok=True)

ckpt_dir = r"/home/wei/NeRF_src/edge-nerf/ckpts/house01/pc"

scene_names = os.listdir(ckpt_dir)
scene_names.sort()

for i, scene_name in enumerate(scene_names):
    print("-" * 50)
    print("processing:", i, ", scene_name:", scene_name)
    ckpt_path = os.path.join(ckpt_dir, scene_name)
    if not os.path.exists(ckpt_path):
        print(ckpt_path + " not exist!")
        continue
    size = 2.0  # default 1.2
    sample_num = 400  #
    sigma, gray = get_sigma_from_nerf(ckpt_path, size=size, N=sample_num)


    sigma_threshold = 0.6
    points = np.argwhere(sigma > sigma_threshold)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    sigma_values = sigma[sigma > sigma_threshold]
    colors = np.zeros((points.shape[0], 3))
    colors[:, 0] = sigma_values
    pcd.colors = o3d.utility.Vector3dVector(colors)

    ply_name = scene_name+'_'+str(size)+'_'+str(sample_num)+'_'+str(sigma_threshold)
    write_path = os.path.join(target_path, ply_name + ".ply")
    o3d.io.write_point_cloud(write_path, pcd, write_ascii=True)     # Set write_ascii to True to output in ascii format, otherwise binary format will be used.

    plt_vis(pcd, save_path=None)    # plot pc
