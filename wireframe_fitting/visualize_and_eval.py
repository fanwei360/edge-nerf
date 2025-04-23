import os
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import random
import json
import torch
import point_cloud_utils as pcu
import open3d as o3d
import pandas as pd


def visualize_pred_gt(all_pred_points, all_gt_points, name, save_fig=False, show_fig=False):
    ax = plt.figure(dpi=120).add_subplot(projection='3d')

    x_pred = [k[0] for k in all_pred_points]
    y_pred = [k[1] for k in all_pred_points]
    z_pred = [k[2] for k in all_pred_points]
    # print("max xyz:", max(x), max(y), max(z))
    ax.scatter(x_pred, y_pred, z_pred, c='r', marker='o', s=0.5, linewidth=0.1, alpha=0.5, cmap='spectral')

    # ---------------------------------plot the gt---------------------------------
    x_gt = [k[0] for k in all_gt_points]
    y_gt = [k[1] for k in all_gt_points]
    z_gt = [k[2] for k in all_gt_points]
    ax.scatter(x_gt, y_gt, z_gt, c='g', marker='o', s=0.5, linewidth=0.1, alpha=0.5, cmap='spectral')


    plt.axis('off')

    ax.view_init(azim=60, elev=60)
    range_size = [0, 1]
    ax.set_zlim3d(range_size[0], range_size[1])
    plt.axis([range_size[0], range_size[1], range_size[0], range_size[1]])
    if save_fig:
        plt.savefig(os.path.join(vis_dir, name + ".png"), bbox_inches='tight')
    if show_fig:
        plt.show()


def sample_points_by_grid(pred_points, num_voxels_per_axis=64):
    normals = pcu.estimate_point_cloud_normals_knn(pred_points, 16)[1]
    bbox_size = np.array([1, 1, 1])
    # The size per-axis of a single voxel
    sizeof_voxel = bbox_size / num_voxels_per_axis
    pred_sampled, _, _ = pcu.downsample_point_cloud_voxel_grid(sizeof_voxel, pred_points, normals)
    pred_sampled = pred_sampled.astype(np.float32)
    return pred_sampled


def get_pred_points(json_path, cube_sample=256, cube_size=1.2, curve_type="cubic", sample_num=100):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    curves_ctl_pts = json_data['curves_ctl_pts']
    num_curves = len(curves_ctl_pts)
    print("num_curves:", num_curves)

    t = np.linspace(0, 1, sample_num)
    if curve_type == "cubic":
        # # -----------------------------------for Cubic Bezier-----------------------------------
        matrix_u = np.array([t ** 3, t ** 2, t, 1], dtype=object)
        matrix_middle = np.array([
            [-1, 3, -3, 1],
            [3, -6, 3, 0],
            [-3, 3, 0, 0],
            [1, 0, 0, 0]
        ])
    elif curve_type == "line":
        # # -------------------------------------for Line-----------------------------------------
        matrix_u = np.array([t, 1], dtype=object)
        matrix_middle = np.array([
            [-1, 1],
            [1, 0]
        ])

    all_points = []
    for i, each_curve in enumerate(curves_ctl_pts):
        each_curve = np.array(each_curve)  # shape: (4, 3)
        each_curve = (each_curve / cube_sample * cube_size*2) - cube_size   # based on the settings of extract point cloud

        matrix = np.matmul(np.matmul(matrix_u.T, matrix_middle), each_curve)
        for i in range(sample_num):
            all_points.append([matrix[0][i], matrix[1][i], matrix[2][i]])

    # exchange X and Y axis, do not know why yet... ...
    all_points = np.array([[pts[1], pts[0], pts[2]] for pts in all_points])
    all_points[:,  1] = -all_points[:,  1]
    return np.array(all_points)

def compute_chamfer_distance(pred_sampled, gt_points, metrics):
    chamfer_dist = pcu.chamfer_distance(pred_sampled, gt_points)
    metrics["chamfer"].append(chamfer_dist)
    print("chamfer_dist:", chamfer_dist)
    return metrics

def compute_precision_recall_IOU(pred_sampled, gt_points, metrics, thresh=0.02):
    dists_a_to_b, _ = pcu.k_nearest_neighbors(pred_sampled, gt_points,
                                              k=1)  # k closest points (in pts_b) for each point in pts_a
    correct_pred = np.sum(dists_a_to_b < thresh)
    precision = correct_pred / len(dists_a_to_b)
    metrics["precision"].append(precision)

    dists_b_to_a, _ = pcu.k_nearest_neighbors(gt_points, pred_sampled, k=1)
    correct_gt = np.sum(dists_b_to_a < thresh)
    recall = correct_gt / len(dists_b_to_a)
    metrics["recall"].append(recall)

    fscore = 2 * precision * recall / (precision + recall)
    metrics["fscore"].append(fscore)

    intersection = min(correct_pred, correct_gt)
    union = len(dists_a_to_b) + len(dists_b_to_a) - max(correct_pred, correct_gt)

    IOU = intersection / union
    metrics["IOU"].append(IOU)
    print("precision:", precision, "recall:", recall, "fscore:", fscore, "IOU:", IOU)
    return metrics

def get_sample_obj_points(obj_path, distance_between_points=0.01):
    """sampling point cloud"""
    vertices = []
    lines = []
    sampled_points = []
    with open(obj_path, 'r') as file:
        obj_file = file.readlines()

    for line in obj_file:
        elements = line.strip().split()
        if elements:
            if elements[0] == 'v':
                vertex_coords = np.array(list(map(float, elements[1:])))
                # vertices.append(map(str, correct_ept))
                vertices.append(vertex_coords)
            elif elements[0] == 'l':
                lines.append(np.array(list(map(int, elements[1:]))))
    for line in lines:
        line = line - 1
        ept0 = vertices[line[0]]
        ept1 = vertices[line[1]]
        line_len = np.linalg.norm(ept0 - ept1)
        sample_nums = int(line_len / distance_between_points)
        t_values = np.linspace(0, 1, sample_nums)
        points = np.array([ept0 + t * (ept1 - ept0) for t in t_values])
        sampled_points.extend(points)
    return np.array(sampled_points).astype(np.float32)


save_curve_dir = r"/home/wei/Datasets/edge-nerf-backup"
gt_dir = os.path.join(save_curve_dir, "gt")

bool_eval = True
save_sample_point = True

result_objs = [each for each in os.listdir(save_curve_dir) if each.endswith(".obj")]
result_objs.sort()
gt_obj_names = [each for each in os.listdir(gt_dir) if each.endswith(".obj")]
csv_path = os.path.join(save_curve_dir, "eval_result.csv")

metrics = {
        "chamfer": [],
        "precision": [],
        "recall": [],
        "fscore": [],
        "IOU": [],
        "name": []
    }


for i, result_name in enumerate(result_objs):
    wf_id = result_name.split('_')[1]
    method_name = result_name.split('_')[0]

    print("-" * 50)
    print("processing:", i, ", name:", result_name)

    # ret_obj_path = os.path.join(save_curve_dir, result_objs[i])
    ret_obj_path = os.path.join(save_curve_dir, result_name)
    pred_obj_points = get_sample_obj_points(ret_obj_path, distance_between_points=0.01)
    pred_sampled = sample_points_by_grid(pred_obj_points)

    if wf_id not in gt_obj_names:
        continue
    gt_path = os.path.join(gt_dir, wf_id)
    gt_obj_points = get_sample_obj_points(gt_path, distance_between_points=0.01)
    gt_sampled = sample_points_by_grid(gt_obj_points)

    if bool_eval:
        metrics = compute_chamfer_distance(pred_sampled, gt_sampled, metrics)
        metrics = compute_precision_recall_IOU(pred_sampled, gt_sampled, metrics, thresh=0.02)
        metrics["name"].append(result_name)
        print("raw preds:", pred_obj_points.shape, ", sampled preds:", pred_sampled.shape, ", gt_raw shape:", gt_obj_points.shape, ", gt shape:", gt_sampled.shape)
        visualize_pred_gt(pred_obj_points, gt_obj_points, result_name, save_fig=False, show_fig=False)

results_df = pd.DataFrame(metrics)
results_df.to_csv(csv_path, index=False)


print("total CADs:", len(result_objs))
print(metrics)
