from __future__ import print_function, division

import argparse
from pathlib2 import Path
import uuid

from mpi4py import MPI
import numpy as np
import scipy.signal as signal
from tqdm import tqdm

from vgn.grasp import Grasp, Label
from vgn.perception import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils import io
from vgn.utils.transform import Rotation, Transform


OBJECT_COUNT_LAMBDA = 4
MAX_VIEWPOINT_COUNT = 6
GRASPS_PER_SCENE = 120


def main(args):
    workers, rank = setup_mpi()
    create_data_dir(args.root, rank)
    sim = ClutterRemovalSim(args.scene, args.object_set, gui=args.sim_gui)
    finger_depth = sim.gripper.finger_depth
    grasps_per_worker = args.num_grasps // workers
    pbar = tqdm(total=grasps_per_worker, disable=rank is not 0)

    for _ in range(grasps_per_worker // GRASPS_PER_SCENE):
        # generate heap
        object_count = np.random.poisson(OBJECT_COUNT_LAMBDA) + 1
        sim.reset(object_count)
        sim.save_state()

        # render synthetic depth images
        depth_imgs, extrinsics = render_images(sim, MAX_VIEWPOINT_COUNT)

        # reconstrct point cloud using a subset of the images
        n = np.random.randint(MAX_VIEWPOINT_COUNT) + 1
        pc = reconstruct_point_cloud(sim, depth_imgs, extrinsics, n)

        # crop surface and borders from point cloud
        pc = pc.crop(sim.lower, sim.upper)
        # o3d.visualization.draw_geometries([pc])

        if pc.is_empty():
            print("Point cloud empty, skipping scene")
            continue

        # store the raw data
        scene_id = store_raw_data(args.root, depth_imgs, extrinsics, n)

        for _ in range(GRASPS_PER_SCENE):
            # sample and evaluate a grasp point
            point, normal = sample_grasp_point(pc, finger_depth)
            grasp, label = evaluate_grasp_point(sim, point, normal)

            # store the sample
            store_sample(args.root, scene_id, grasp, label)
            pbar.update()

    pbar.close()


def setup_mpi():
    workers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return workers, rank


def create_data_dir(root, rank):
    if rank != 0:
        return
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = root / "grasps.csv"
    if not csv_path.exists():
        io.create_csv(
            csv_path,
            ["scene_id", "qx", "qy", "qz", "qw", "x", "y", "z", "width", "label"],
        )


def render_images(sim, N):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0])

    extrinsics = np.empty((N, 7), np.float32)
    depth_imgs = np.empty((N, height, width), np.float32)

    for i in range(N):
        r = np.random.uniform(1.6, 2.4) * sim.size
        theta = np.random.uniform(0.0, np.pi / 4.0,)
        phi = np.random.uniform(0.0, 2.0 * np.pi)

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        depth_img = sim.camera.render(extrinsic)[1]

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img

    return depth_imgs, extrinsics


def reconstruct_point_cloud(sim, depth_imgs, extrinsics, n):
    tsdf = TSDFVolume(sim.size, 120)
    for i in range(n):
        extrinsic = Transform.from_list(extrinsics[i])
        tsdf.integrate(depth_imgs[i], sim.camera.intrinsic, extrinsic)
    return tsdf.extract_point_cloud()


def sample_grasp_point(point_cloud, finger_depth, eps=0.1):
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    ok = False
    while not ok:
        # TODO this could result in an infinite loop, though very unlikely
        idx = np.random.randint(len(points))
        point, normal = points[idx], normals[idx]
        ok = normal[2] > -0.1  # make sure the normal is poitning upwards
    grasp_depth = np.random.uniform(-eps * finger_depth, (1.0 + eps) * finger_depth)
    point = point + normal * grasp_depth
    return point, normal


def evaluate_grasp_point(sim, pos, normal, num_rotations=6):
    # define initial grasp frame on object surface
    z_axis = -normal
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_dcm(np.vstack((x_axis, y_axis, z_axis)).T)

    # try to grasp with different yaw angles
    yaws = np.linspace(0.0, np.pi, num_rotations)
    outcomes, widths = [], []
    for yaw in yaws:
        ori = R * Rotation.from_euler("z", yaw)
        sim.restore_state()
        candidate = Grasp(Transform(ori, pos), width=sim.gripper.max_opening_width)
        outcome, width = sim.execute_grasp(candidate, remove=False)
        outcomes.append(outcome)
        widths.append(width)

    # detect mid-point of widest peak of successful yaw angles
    # TODO currently this does not properly handle periodicity
    successes = (np.asarray(outcomes) == Label.SUCCESS).astype(float)
    if np.sum(successes):
        peaks, properties = signal.find_peaks(
            x=np.r_[0, successes, 0], height=1, width=1
        )
        idx_of_widest_peak = peaks[np.argmax(properties["widths"])] - 1
        ori = R * Rotation.from_euler("z", yaws[idx_of_widest_peak])
        width = widths[idx_of_widest_peak]

    return Grasp(Transform(ori, pos), width), int(np.max(outcomes))


def store_raw_data(root, depth_imgs, extrinsics, n):
    scene_id = uuid.uuid4().hex
    path = root / "raw" / (scene_id + ".npz")
    np.savez_compressed(str(path), depth_imgs=depth_imgs, extrinsics=extrinsics, n=n)
    return scene_id


def store_sample(root, scene_id, grasp, label):
    # add a row to the table (TODO concurrent writes could be an issue?)
    csv_path = root / "grasps.csv"
    qx, qy, qz, qw = grasp.pose.rotation.as_quat()
    x, y, z = grasp.pose.translation
    width = grasp.width
    io.append_csv(csv_path, scene_id, qx, qy, qz, qw, x, y, z, width, label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--scene", type=str, choices=["pile", "packed"])
    parser.add_argument("--object-set", type=str)
    parser.add_argument("--num-grasps", type=int, default=10000)
    parser.add_argument("--sim-gui", action="store_true")
    args = parser.parse_args()
    main(args)