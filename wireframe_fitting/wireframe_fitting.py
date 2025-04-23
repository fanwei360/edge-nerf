import os
import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import time
import json
import point_cloud_utils as pcu
import concurrent.futures
from collections import defaultdict

from ChamferDistancePytorch import fscore
from ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
# https://github.com/ThibaultGROUEIX/ChamferDistancePytorch


def chamfer_example():
    chamLoss = dist_chamfer_3D.chamfer_3DDist()
    points1 = torch.rand(32, 1000, 3).cuda()
    points2 = torch.rand(32, 2000, 3, requires_grad=True).cuda()
    dist1, dist2, idx1, idx2 = chamLoss(points1, points2)
    print(dist1.shape, dist2.shape, idx1.shape, idx2.shape)
    f_score, precision, recall = fscore.fscore(dist1, dist2)
    print(f_score.shape, precision.shape, recall.shape)

    chamfer_loss = torch.sqrt(dist1).mean() + torch.sqrt(dist2).mean()
    print(chamfer_loss)
    # print(idx1)
    # print(idx2)
    return 0

def compute_density(index, pcd, kdtree, radius):
    _, indices, _ = kdtree.search_radius_vector_3d(pcd.points[index], radius)
    return len(indices)


class Curves_Model(nn.Module):
    def __init__(self, n_curves=12, initial_params=None, curve_type="cubic"):
        super(Curves_Model, self).__init__()
        self.curve_type = curve_type
        if self.curve_type == "cubic":
            self.n_ctl_points = 4
            self.matrix_w = torch.tensor([
                [-1, 3, -3, 1],
                [3, -6, 3, 0],
                [-3, 3, 0, 0],
                [1, 0, 0, 0]
            ]).float().cuda()
        elif self.curve_type == "line":
            self.n_ctl_points = 2
            self.matrix_w = torch.tensor([
                [-1, 1],
                [1, 0]
            ]).float().cuda()
        self.matrix_t = self.get_matrix_t(num=100)

        if initial_params is None:
            params = torch.rand(n_curves, self.n_ctl_points, 3, requires_grad=True).cuda()  # n * 4(2) * 3；
        else:
            params = initial_params.cuda()
        assert params.shape == (n_curves, self.n_ctl_points, 3)
        self.params = nn.Parameter(params)

    def initialize_params_center(self, pts_target, radius, init_pts):
        print("Initializing parameters... ...")
        init_mode = "density"
        if init_mode == "center":
            self.max_density = 25
            center_pts = torch.mean(pts_target.squeeze(), axis=0)
            print('Initialize the endpoints：', center_pts)
            self.params.requires_grad = False
            for i in range(len(self.params)):
                for j in range(len(self.params[i])):
                    self.params[i][j] = center_pts
            self.params.requires_grad = True
        elif init_mode == "density":
            self.params.requires_grad = False
            pts_target_cpu = pts_target.detach().cpu().numpy()
            for i in range(len(pts_target_cpu)):
                pts = pts_target_cpu[i]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                kdtree = o3d.geometry.KDTreeFlann(pcd)

                densities = np.zeros(len(pts))
                for k in range(len(pts)):
                    [_, indices, _] = kdtree.search_radius_vector_3d(pcd.points[k], radius)
                    densities[k] = len(indices)
                max_density_point = pts[np.argmax(densities)]
                self.max_density = densities[np.argmax(densities)]
                print('max density：', densities[np.argmax(densities)])
                for j in range(len(self.params[i])):
                    if j % 2 == 0:
                        self.params[i][j] = torch.tensor(max_density_point).to(self.params.device)
                    else:
                        max_density_point_ = max_density_point.copy()
                        max_density_point_[-1] += 3
                        self.params[i][j] = torch.tensor(max_density_point_).to(self.params.device)
            self.params.requires_grad = True
        elif init_mode == "farthest":
            self.max_density = 20
            print('Initialize the endpoints：', init_pts)
            distances = torch.norm(pts_target - init_pts, dim=-1)
            farthest_point_index = torch.argmax(distances)
            farthest_point = pts_target[:, farthest_point_index, :]
            self.params.requires_grad = False
            for i in range(len(self.params)):
                for j in range(len(self.params[i])):
                    self.params[i][j] = farthest_point
            self.params.requires_grad = True
            return farthest_point

    def get_matrix_t(self, num=50):
        matrix_t = []
        if self.curve_type == "cubic":
            for t in np.linspace(0, 1, num):
                each_matrix_t = torch.tensor([
                    t * t * t,
                    t * t,
                    t,
                    1
                ])
                matrix_t.append(each_matrix_t)
        elif self.curve_type == "line":
            for t in np.linspace(0, 1, num):
                each_matrix_t = torch.tensor([
                    t,
                    1
                ])
                matrix_t.append(each_matrix_t)

        matrix_t = torch.stack(matrix_t, axis=0).float().cuda()
        return matrix_t

    def forward(self, var_coef=1):
        # else:
        matrix1 = torch.einsum('ik,kj->ij', [self.matrix_t, self.matrix_w])  # shape: [100, 4(2)] * [4, 4(2)] = [100, 4(2)]
        matrix2 = torch.einsum('ik,nkj->nij', [matrix1, self.params])  # shape: [100, 4(2)] * [n, 4(2), 3] = [n, 100, 3]
        pts_curve = matrix2.reshape(1, -1, 3)  # shape: [1, n * 100, 3]

        multiply = 5        # default 5
        pts_curve_m = pts_curve.repeat(1, multiply, 1)  # shape: [1, n * 100 * multiply, 3]
        noise = torch.randn_like(pts_curve_m)
        variance = 0.5      # default 0.5
        noise = (variance ** 0.5) * noise * var_coef
        # print(torch.mean(noise), torch.max(noise), torch.min(noise))
        pts_curve_m = pts_curve_m + noise

        return pts_curve, pts_curve_m, self.params


def optimize_one_line(max_iters, pts_target, dist_pts, density_radius, farthest_point, alpha=5, curve_type="cubic"):

    chamLoss = dist_chamfer_3D.chamfer_3DDist()
    curve_model = Curves_Model(n_curves=1, curve_type=curve_type)
    if farthest_point is None:
        farthest_point = pts_target[:, 0, :]
    farthest_point = curve_model.initialize_params_center(pts_target, density_radius, farthest_point)
    # print(curve_model.params)

    lr = 0.5
    optimizer = torch.optim.Adam(curve_model.parameters(), lr=lr)

    for iters in range(max_iters):
        pts_curve, pts_curve_m, current_params = curve_model()

        # chamfer loss
        dist1, dist2, idx1, idx2 = chamLoss(pts_curve_m, pts_target)

        chamfer_loss_1 = alpha * torch.sqrt(dist1).mean()
        chamfer_loss_2 = torch.sqrt(dist2).mean()
        loss_chamfer = chamfer_loss_1 + chamfer_loss_2

        # dist_var = 0
        # if iters >= 200:
        #     # 点到直线的投影距离方差，保证拟合均匀性
        #     delete_idx = idx1[dist1 < dist_pts].unique().to(torch.long)
        #     pts_to_delete = pts_target[:, delete_idx, :]
        #     distAB = current_params[:, 1, :] - current_params[:, 0, :]
        #     PAt = torch.sum((pts_to_delete - current_params[:, 0, :]) * distAB, dim=-1)/torch.sum(distAB**2, dim=-1)
        #     integers = torch.tensor([[0, 1]], dtype=PAt.dtype, device=PAt.device)
        #     PAt, _ = torch.sort(torch.cat((PAt, integers), dim=-1).view(-1, 1), dim=0)
        #     posPA = current_params[:, 0, :] + PAt * distAB
        #     distance = torch.norm(posPA[1:] - posPA[:-1], dim=1)
        #     dist_var = torch.var(distance)

        loss = loss_chamfer # + 5 * dist_var

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"iters: {iters}, loss_total: {loss}, chamfer_loss: {chamfer_loss_1}, alpha: {alpha}")
    return current_params, pts_curve, pts_curve_m, curve_model.max_density, farthest_point


def updata_pts_target(pts_curve, pts_target, distance):
    # delete point cloud close to current curve
    pts_curve = pts_curve.squeeze(0).cpu().detach().numpy()
    pts_target = pts_target.squeeze(0).cpu().detach().numpy()
    print("pts_curve.shape:", pts_curve.shape, "pts_target.shape:", pts_target.shape)

    # distance = 4    # default 4
    dists_a_to_b, corrs_a_to_b = pcu.k_nearest_neighbors(pts_curve, pts_target, k=100)
    delete_index = corrs_a_to_b[dists_a_to_b < distance]
    delete_index = list(set(delete_index))
    print("Deleting " + str(len(delete_index)) + " points")
    pts_delete = pts_target[delete_index]
    pts_target = np.delete(pts_target, delete_index, axis=0)
    # pts_target = torch.from_numpy(pts_target).float().unsqueeze(0).cuda()

    return pts_target, len(delete_index), pts_delete


def Line2Cubic(curves_ctl_pts):
    curves_ctl_pts_new = []
    for each_curve in curves_ctl_pts:
        each_curve = np.array(each_curve)
        extra_pts1 = 2 / 3 * each_curve[0] + 1 / 3 * each_curve[1]
        extra_pts2 = 1 / 3 * each_curve[0] + 2 / 3 * each_curve[1]
        new_curve = np.array([each_curve[0], extra_pts1, extra_pts2, each_curve[1]]).tolist()
        curves_ctl_pts_new.append(new_curve)
    return curves_ctl_pts_new

def get_sameline_idx(n):
    """ Excluding distances to endpoints of the same line"""
    sequence = [0]
    for i in range(1, n):
        value = sequence[-1] + (2 * n - i)
        sequence.append(value)
    for i in range(len(sequence)):
        sequence[i] += n - 1
    return sequence

def get_line_idx(m, dist_idx, start_pts_idx):
    """
    input：Distance index in dists
    output：Linear index corresponding to distance index
    """

    common_elements = torch.tensor(np.intersect1d(start_pts_idx.cpu().numpy(), dist_idx.cpu().numpy())).to(dist_idx.device)
    if common_elements.shape[0] > 0:
        common_idx = torch.where(start_pts_idx.unsqueeze(0) == common_elements.unsqueeze(1))[1]
        common_idx_ = torch.where(dist_idx.unsqueeze(0) == common_elements.unsqueeze(1))[1]
        # Remove duplicate elements from dist_idx
        mask = torch.ones(len(dist_idx), dtype=torch.bool)
        mask[common_idx_] = False
        dist_idx = dist_idx[mask]
        near_line_idx1 = torch.stack((common_idx % (m/2), common_idx % (m/2)+1), dim=1)

    lineA_idx = torch.searchsorted(start_pts_idx, dist_idx) - 1
    move_dist = dist_idx - start_pts_idx[lineA_idx]
    lineB_idx = lineA_idx + move_dist + 1
    lineA_idx_ = lineA_idx % (m / 2)
    lineB_idx_ = lineB_idx % (m / 2)
    near_line_idx2 = torch.cat((lineA_idx_.unsqueeze(1), lineB_idx_.unsqueeze(1)), dim=1)
    if common_elements.shape[0] > 0:
        near_line_idx = torch.cat((near_line_idx1, near_line_idx2)).long()
    else:
        near_line_idx = near_line_idx2.long()
    filtered_lines = near_line_idx[(near_line_idx[:, 0] < m/2) & (near_line_idx[:, 1] < m/2)]
    if filtered_lines.max() >= m/2:
        print("error")
    return filtered_lines

def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def merge_duplicate_points(points, distance_threshold):
    """Merging line segment endpoints"""
    point_dict = defaultdict(list)
    merged_points = []
    end_points = points.reshape(-1, 3)
    point_idx_dict = defaultdict(list)
    edges = []

    for i, point in enumerate(end_points):
        merged = False
        for key in point_dict:
            for existing_point in point_dict[key]:
                if calculate_distance(point, existing_point) < distance_threshold:
                    point_dict[key].append(point)
                    point_idx_dict[key].append(i)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            new_key = len(point_dict)
            point_dict[new_key].append(point)
            point_idx_dict[new_key].append(i)

    for key in point_dict:
        merged_points.append(np.mean(point_dict[key], axis=0))

    for i in point_idx_dict:
        for old_idx in point_idx_dict[i]:
            if old_idx % 2 == 0:
                for j in point_idx_dict:
                    if old_idx+1 in point_idx_dict[j]:
                        edges.append(sorted([i, j]))
                        break
    #  De-duplication
    unique_edges = []
    [unique_edges.append(item) for item in edges if item not in unique_edges and item[0] != item[1]]
    return merged_points, unique_edges


def calculate_angle(vector1, vector2):
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector1, unit_vector2)
    return np.arccos(np.clip(dot_product, -1.0, 1.0))

def angle_filter_edges(merged_points, unique_edges, angle_threshold=10*np.pi/180.0):
    """Filtering line segments based on angle"""
    # point-to-edge mappings
    point_to_edges = {i: [] for i in range(len(merged_points))}
    for edge in unique_edges:
        if edge[0] != edge[1]:
            point_to_edges[edge[0]].append(edge)
            point_to_edges[edge[1]].append(edge)

    # Points and edges to be deleted
    points_to_remove = set()
    edges_to_remove = set()

    for point_index, connected_edges in point_to_edges.items():
        if len(connected_edges) != 1:
            continue
        single_edge_point = point_index
        connected_point = list(filter(lambda x: x != single_edge_point, connected_edges[0]))[0]

        other_point_edges = point_to_edges[connected_point]
        other_connected_edges = list(filter(lambda x: x != connected_edges[0], other_point_edges))
        vector1 = merged_points[single_edge_point] - merged_points[connected_point]
        for edge in other_connected_edges:
            other_point = list(filter(lambda x: x != connected_point, edge))[0]
            vector2 = merged_points[other_point] - merged_points[connected_point]
            angle = calculate_angle(vector1, vector2)

            if angle < angle_threshold:
                edges_to_remove.add(tuple(connected_edges[0]))
                points_to_remove.add(single_edge_point)
                break

    unique_edges_ = [edge for edge in unique_edges if tuple(edge) not in edges_to_remove]
    merged_points_ = [point for i, point in enumerate(merged_points) if i not in points_to_remove]
    unique_edges_arr = np.array(unique_edges_)
    for point_idx in sorted(points_to_remove, reverse=True):
        unique_edges_arr[unique_edges_arr >= point_idx] -= 1

    print("Angle Filter Line：\n", len(edges_to_remove))
    print("Remaining line segments：\n", len(unique_edges_))
    return merged_points_, list(unique_edges_arr)

def save_obj(file_path, vertices, edges):
    with open(file_path, 'w') as file:
        for vertex in vertices:
            file.write(f'v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n')
        for edge in edges:
            file.write(f'l {edge[0] + 1} {edge[1] + 1}\n')

def translation_edps(curves_params, d):
    # current_params = curves_params.detach().cpu().numpy()
    direction = (curves_params[0, 1] - curves_params[0, 0])
    unit_direction = direction / torch.linalg.norm(direction)
    translation = unit_direction * (d / 2) # fallback endpoint

    new_params = curves_params.clone()
    new_params[0, 0] = curves_params[0, 0] + translation
    new_params[0, 1] = curves_params[0, 1] - translation

    matrix_w = torch.tensor([
        [-1, 1],
        [1, 0]
    ]).float().cuda()

    matrix_t = []
    for t in np.linspace(0, 1, num=100):
        each_matrix_t = torch.tensor([
            t,
            1
        ])
        matrix_t.append(each_matrix_t)
    matrix_t = torch.stack(matrix_t, axis=0).float().cuda()

    matrix1 = torch.einsum('ik,kj->ij', [matrix_t, matrix_w])
    matrix2 = torch.einsum('ik,nkj->nij', [matrix1, new_params])
    pts_curve = matrix2.reshape(1, -1, 3)  # shape: [1, n * 100, 3]

    return pts_curve

def correct_coordinate(obj_path, cube_sample=256, cube_size=1.2):
    """Recovery coordinates"""
    directory, filename = os.path.split(obj_path)
    output_obj_file = os.path.join(directory, filename.replace('.obj', '_cor.obj'))
    vertices = []
    lines = []
    rotation_matrix = np.array([[0, 1, 0],
                                [-1, 0, 0],
                                [0, 0, 1]])

    with open(obj_path, 'r') as file:
        obj_file = file.readlines()

    for line in obj_file:
        elements = line.strip().split()
        if elements:
            if elements[0] == 'v':
                vertex_coords = np.array(list(map(float, elements[1:])))
                correct_ept = (vertex_coords / cube_sample * cube_size * 2) - cube_size
                correct_ept = np.matmul(rotation_matrix, correct_ept)   # to bender
                correct_ept[1], correct_ept[2] = correct_ept[2], -correct_ept[1]
                # vertices.append(map(str, correct_ept))
                vertices.append(correct_ept)
            elif elements[0] == 'l':
                lines.append(elements[1:])

    with open(output_obj_file, 'w') as file:
        for vertex in vertices:
            file.write(f"v {' '.join(map(str,vertex))}\n")
        for line in lines:
            file.write(f"l {' '.join(map(str, line))}\n")
    print("output to", output_obj_file)


def line_to_line_dist(x,y):
    dis = ((x[:,None,:, None] - y[:,None])**2).sum(-1)
    dis = np.sqrt(dis)
    dis = np.minimum(
        dis[:, :, 0, 0] + dis[:, :, 1, 1], dis[:, :, 0, 1] + dis[:, :, 1, 0]
    )
    return dis

def lineNms(line, line_nms_thresh=20):
    dis = line_to_line_dist(line,line)
    dropped_line_index = []
    thresh = line_nms_thresh
    for i in range(dis.shape[0]):
        if i in dropped_line_index:
            continue
        d = dis[i]
        same_line_indexes = (d<thresh).nonzero()[0]
        for same_line_index_i in same_line_indexes:
            if same_line_index_i == i:
                continue
            else:
                dropped_line_index.append(same_line_index_i)
    line_removal = np.delete(line, dropped_line_index, axis=0)
    print("original line segments：\n", line.shape[0])
    print("Remaining line segments：\n", line_removal.shape[0])
    return line_removal

if __name__ == '__main__':
    # chamfer_example()
    print("->Loading Point Cloud... ...")

    point_cloud_dir = r"/home/wei/NeRF_src/edge-nerf/pointcloud/house01"
    save_curve_dir = r"/home/wei/NeRF_src/edge-nerf/wireframe"
    os.makedirs(save_curve_dir, exist_ok=True)

    scene_names = os.listdir(point_cloud_dir)
    scene_names.sort()
    only_stage2 = False

    dist_pts = 6
    density_radius = dist_pts/2
    nms_thresh = dist_pts * 2
    del_num = 40
    alpha = 5

    for i, scene_name in enumerate(scene_names):
        cube_size = float(scene_name.split('_')[-3])
        cube_sample = float(scene_name.split('_')[-2])
        print("-" * 50)
        print("processing:", i, ", scene_name:", scene_name)
        pcd_path = os.path.join(point_cloud_dir, scene_name)
        pcd = o3d.io.read_point_cloud(pcd_path)
        init_pts_target = torch.from_numpy(np.asarray(pcd.points)).float().unsqueeze(0).cuda()
        print("initial pts_target.shape:", init_pts_target.shape)
        curve_type = "line"

        if not only_stage2:
            # # stage 1: ------------------------per curve optimization-------------------------
            print("Ready to conduct stage 1 ... ...")
            start = time.perf_counter()
            pts_target = init_pts_target.clone()
            raw_pts = init_pts_target.clone()
            cur_curves = []

            farthest_point = None
            for i in range(1000):
                print('=' * 70)
                curves_params, pts_curve, pts_curve_m, max_denstiy, farthest_point = optimize_one_line(max_iters=400, pts_target=pts_target, density_radius=density_radius,
                                                                                                       dist_pts=dist_pts, farthest_point=farthest_point, alpha=alpha, curve_type=curve_type)
                pts_curve_ = translation_edps(curves_params, dist_pts)
                pts_target, delete_num, pts_delete = updata_pts_target(pts_curve_, pts_target, dist_pts)
                len_line = torch.norm(curves_params[:, 0] - curves_params[:, 1], dim=-1)

                if delete_num > del_num and len_line > dist_pts:
                    cur_curves.append(np.array(curves_params.detach().cpu()))

                print("Current pts_target.shape:", pts_target.shape)
                print('Current number of Curves:', len(cur_curves))
                if pts_target.shape[0] < 20 or max_denstiy < 10:   # if there are very few points, stop the optimazation
                    break
                pts_target = torch.from_numpy(pts_target).float().unsqueeze(0).cuda()

            cur_curves = np.array(cur_curves).squeeze(1)    # (total_curves, 1, 4, 3) to (total_curves, 4, 3)
            print("total curves:", cur_curves.shape)
            print("Total time comsumed", time.perf_counter() - start)

            json_data = {
                "date": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                "scene_name": scene_name,
                'curves_ctl_pts': cur_curves.tolist()
            }
            file_name = "record_" + scene_name[:-4] + "_stage1_" + curve_type + ".json"
            json_path = os.path.join(save_curve_dir, file_name)
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(json_data, f)
            print("json file saved in", json_path)
            merged_points_1, edges_1 = merge_duplicate_points(cur_curves, 0.0)
            # obj_path = json_path.rsplit('.', 1)[0] + '.obj'
            obj_path = os.path.join(save_curve_dir, file_name.replace('.json', '.obj'))
            save_obj(obj_path, merged_points_1, edges_1)
            print("obj file saved in", obj_path)

        # stage 2:------------------------all curve refinement-------------------------
        print("Ready to conduct stage 2 ... ...")
        time.sleep(1)
        file_name = "record_" + scene_name[:-4] + "_stage1_" + curve_type + ".json"
        json_path = os.path.join(save_curve_dir, file_name)
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        curves_ctl_pts = json_data['curves_ctl_pts']
        # print(curves_ctl_pts)
        # print(torch.tensor(curves_ctl_pts).shape)
        print("Number of curves:", len(curves_ctl_pts))

        start = time.perf_counter()
        chamLoss = dist_chamfer_3D.chamfer_3DDist()

        curve_type = "line"
        curve_model = Curves_Model(n_curves=len(curves_ctl_pts), initial_params=torch.tensor(curves_ctl_pts), curve_type=curve_type)

        # Excluding distances to endpoints of the same line
        num_line = len(curves_ctl_pts)
        lpt_idx = get_sameline_idx(num_line)

        # Index of points in dists
        num_epts = num_line*2
        start_pts_idx = [0]
        for i in range(1, num_epts):
            start_pts_idx.append(start_pts_idx[i - 1] + (num_epts - i))
        start_pts_idx.pop()
        start_pts_idx = torch.tensor(start_pts_idx).to(curve_model.params.device)

        lr = 0.5
        optimizer = torch.optim.Adam(curve_model.parameters(), lr=lr)
        stand_dir = torch.tensor([1, -1, 0]).to(curve_model.params.device)
        for iters in range(1000):
            pts_curve, pts_curve_m, current_params = curve_model()

            # calculate endpoints loss
            end_pts = torch.cat([current_params[:, 0, :], current_params[:, -1, :]], dim=0)  # ABCabc
            dists = torch.pdist(end_pts, p=2)       # torch.Size([(n * 2) ** 2 / 2 - n])
            mask = torch.ones_like(dists)
            mask[dists > dist_pts] = 0
            mask[lpt_idx] = 0
            masked_dists = dists * mask
            loss_end_pts = 0.01 * masked_dists.sum()

            line_dir = current_params[:, 1, :] - current_params[:, 0, :]
            unit_dir = line_dir / torch.norm(line_dir, dim=1, keepdim=True)

            # Manhattan assumption
            adjacent_dist_idx = torch.nonzero(masked_dists).flatten()   # Get the index of the nearest neighbor endpoints in the dists
            near_line_idx = get_line_idx(num_epts, adjacent_dist_idx, start_pts_idx)
            dir_A = unit_dir[near_line_idx[:, 0]]
            dir_B = unit_dir[near_line_idx[:, 1]]
            inner_products = torch.sum(dir_A * dir_B, dim=1)
            expand_inner_products = torch.unsqueeze(inner_products, dim=1).expand(-1, 3)
            min_diff, _ = torch.min(torch.abs(expand_inner_products - stand_dir), dim=1)
            manhattan_loss = torch.mean(min_diff)

            # chamfer loss
            dist1, dist2, idx1, idx2 = chamLoss(pts_curve_m, init_pts_target)

            alpha = 1
            chamfer_loss_1 = alpha * torch.sqrt(dist1).mean()
            chamfer_loss_2 = torch.sqrt(dist2).mean()
            loss = chamfer_loss_1 + chamfer_loss_2 + loss_end_pts + manhattan_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (iters + 1) % 100 == 0:
                print(f"iters: {iters}, loss_total: {loss}, loss_1: {chamfer_loss_1}, loss_end_pts: {loss_end_pts}, alpha: {alpha}，loss_manhattan: {manhattan_loss}")

        print("Stage 2 time comsumed", time.perf_counter() - start)

        json_data = {
            "date": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "scene_name": scene_name,
            'curves_ctl_pts': current_params.tolist()
        }
        file_name = "record_" + scene_name[:-4] + "_stage2_" + curve_type + ".json"
        json_path = os.path.join(save_curve_dir, file_name)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(json_data, f)
        print("json file saved in", json_path)

        end_points = current_params.detach().cpu().numpy()
        # NMS
        end_points = lineNms(end_points, nms_thresh)
        merged_points, edges = merge_duplicate_points(end_points, dist_pts)
        merged_points, edges = angle_filter_edges(merged_points, edges)
        # Save the merged line segment as OBJ
        obj_path = os.path.join(save_curve_dir, file_name.replace('.json', '.obj'))
        save_obj(obj_path, merged_points, edges)
        print("obj file saved in", obj_path)

        # Coordinate correction
        correct_coordinate(obj_path, cube_sample, cube_size)
