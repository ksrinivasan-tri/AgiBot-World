import h5py
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import Union
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imageio.v3 as iio
import av
import json
import open3d as o3d
import argparse

HEAD_COLOR = "head_color.mp4"
HAND_LEFT_COLOR = "hand_left_color.mp4"
HAND_RIGHT_COLOR = "hand_right_color.mp4"
HEAD_CENTER_FISHEYE_COLOR = "head_center_fisheye_color.mp4"
HEAD_LEFT_FISHEYE_COLOR = "head_left_fisheye_color.mp4"
HEAD_RIGHT_FISHEYE_COLOR = "head_right_fisheye_color.mp4"
BACK_LEFT_FISHEYE_COLOR = "back_left_fisheye_color.mp4"
BACK_RIGHT_FISHEYE_COLOR = "back_right_fisheye_color.mp4"
HEAD_DEPTH = "head_depth"

def get_rainbow_color(index, max_index):
    cmap = plt.cm.rainbow
    normalized_index = index / max_index
    return cmap(normalized_index)[:3]  # RGB only

def get_transformation_matrix(extrinsic_params):
    rotation_matrix = np.array(extrinsic_params['rotation_matrix'])
    translation_vector = np.array(extrinsic_params['translation_vector']).reshape(3, 1)
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = translation_vector.flatten()
    extrinsic_matrix = np.linalg.inv(extrinsic_matrix)  # Invert to get camera pose
    return extrinsic_matrix

    
def visualize_pointcloud_with_end_effector(
    rgb_frames, depth_frames, end_effector_pos, intrinsic, extrinsic_params, num_future=16
):
    """
    Visualizes an animation of point clouds with end-effector positions in Open3D.

    Parameters:
        rgb_frames (list): List of RGB images (H x W x 3).
        depth_frames (list): List of depth images (H x W).
        end_effector_pos (np.ndarray): End-effector positions (N, 2, 3), where N is the number of frames.
        intrinsic (o3d.camera.PinholeCameraIntrinsic): Camera intrinsic parameters.
        num_future (int): Number of future end-effector positions to visualize.
    """
    def get_rainbow_color(index, max_index):
        cmap = plt.cm.rainbow
        normalized_index = index / max_index
        return cmap(normalized_index)[:3]  # RGB only

    # Create extrinsic transformation matrix
    rotation_matrix = np.array(extrinsic_params['rotation_matrix'])
    translation_vector = np.array(extrinsic_params['translation_vector']).reshape(3, 1)
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = translation_vector.flatten()

    extrinsic_matrix = np.linalg.inv(extrinsic_matrix)  # Invert to get camera pose

    # Initialize visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Visualize until len(rgb_frames) - num_future
    max_frames = len(rgb_frames) - num_future

    for frame_idx in range(max_frames):
        # Create RGBD image for current frame
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_frames[frame_idx]),
            o3d.geometry.Image(depth_frames[frame_idx]["observation.images.cam_top_depth"]),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False,
        )

        # Create point cloud
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        vis.add_geometry(point_cloud)
        # vis.add_geometry(left_sphere)
        # vis.add_geometry(right_sphere)

        # Add future end-effector positions
        future_spheres = []
        for future_idx in range(1, num_future + 1):  # Future positions
            future_frame_idx = frame_idx + future_idx
            if future_frame_idx < len(end_effector_pos):
                future_left = end_effector_pos[future_frame_idx]["state_position"][0]
                future_right = end_effector_pos[future_frame_idx]["state_position"][1]

                # Create spheres for future positions
                future_left_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
                future_left_sphere.translate(future_left)
                future_left_sphere.paint_uniform_color(get_rainbow_color(future_idx, num_future))

                future_right_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
                future_right_sphere.translate(future_right)
                future_right_sphere.paint_uniform_color(get_rainbow_color(future_idx, num_future))

                vis.add_geometry(future_left_sphere)
                vis.add_geometry(future_right_sphere)

                # Collect for removal
                future_spheres.extend([future_left_sphere, future_right_sphere])

        # Update the visualization and capture the frame
        vis.poll_events()
        vis.update_renderer()

        # Remove previous frame's point cloud and spheres
        vis.remove_geometry(point_cloud)
        vis.remove_geometry(left_sphere)
        vis.remove_geometry(right_sphere)
        for sphere in future_spheres:
            vis.remove_geometry(sphere)

    vis.destroy_window()

def undistort_image_with_remap(image, intrinsic_params):
    """
    Undistort an image using remap for better artifact handling.

    Parameters:
        image (np.ndarray): Input RGB or depth image.
        intrinsic_params (dict): Dictionary containing intrinsic and distortion parameters.

    Returns:
        np.ndarray: Undistorted image.
    """
    # Intrinsic matrix
    K = np.array([
        [intrinsic_params['fx'], 0, intrinsic_params['ppx']],
        [0, intrinsic_params['fy'], intrinsic_params['ppy']],
        [0, 0, 1]
    ])

    # Distortion coefficients
    dist_coeffs = np.array([
        intrinsic_params['k1'], intrinsic_params['k2'], intrinsic_params['p1'],
        intrinsic_params['p2'], intrinsic_params['k3']
    ])

    # Image size
    h, w = image.shape[:2]

    # Optimal camera matrix
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))

    # Precompute remap
    map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeffs, None, new_K, (w, h), cv2.CV_32FC1)
    undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return undistorted_image

def create_intrinsic_matrix(intrinsic_params):
    """
    Create a 3x3 intrinsic matrix from given intrinsic parameters.

    Parameters:
        intrinsic_params (dict): Dictionary containing fx, fy, ppx, and ppy.

    Returns:
        np.ndarray: 3x3 camera intrinsic matrix.
    """
    fx = intrinsic_params['fx']
    fy = intrinsic_params['fy']
    ppx = intrinsic_params['ppx']
    ppy = intrinsic_params['ppy']

    intrinsic_matrix = np.array([
        [fx, 0,  ppx],
        [0,  fy, ppy],
        [0,  0,  1]
    ])

    return intrinsic_matrix

# Utility: Load video frames
def load_video_frames(video_path, fps=30):
    """Load frames from a video file."""
    video_frames = []
    video_reader = iio.get_reader(video_path, plugin="pyav")
    for frame in video_reader:
        video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    return video_frames

DEFAULT_IMAGE_PATH = (
    "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.jpg"
)

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    # x_ = T.ToTensor()(x_) # (3, H, W)
    return x_

def create_rgb_depth_video(rgb_frames, frames, output_path, clip_value=10.0, fps=60):
    """
    Creates an MP4 video visualizing RGB and depth images side by side.

    Parameters:
        rgb_frames (list or array): List or array of RGB images.
        frames (list or dict): List or dict containing depth images.
        output_path (str): Path to save the output video file.
        clip_value (float): Maximum depth value for clipping.
        fps (int): Frames per second for the output video.
    """
    # Get the shape of the first frame to define the video size
    rgb_height, rgb_width, _ = rgb_frames[0].shape
    depth_height, depth_width = frames[0]["observation.images.cam_top_depth"].shape

    # Ensure RGB and depth images are of the same height
    assert rgb_height == depth_height, "RGB and Depth images must have the same height."

    # Define video writer
    video_size = (rgb_width + depth_width, rgb_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, video_size)

    for i, (rgb_image, frame) in enumerate(zip(rgb_frames, frames)):
        # Extract and process depth image
        depth_image = frame["observation.images.cam_top_depth"]
        depth_clipped = np.clip(depth_image, 0, clip_value)

        depth_colored = visualize_depth(depth_clipped)
        # depth_normalized = (depth_clipped / clip_value * 255).astype(np.uint8)
        # depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        rgb_image = rgb_image[:, :, ::-1]  # Convert RGB to BGR for OpenCV
        depth_colored = np.array(depth_colored)[:, :, ::-1]  # Convert RGB to BGR for OpenCV
        # Concatenate RGB and depth images
        combined_image = np.hstack((rgb_image, depth_colored))

        # Write the frame to the video
        out.write(combined_image)

    # Release the video writer
    out.release()
    print(f"Video saved to {output_path}")


def load_camera_parameters(base_path):
    """
    Load extrinsic and intrinsic parameters for all non-fisheye cameras.

    Parameters:
        base_path (str): Path to the directory containing camera parameter files.

    Returns:
        dict: Dictionary with camera names as keys and their parameters as values.
    """
    base_path = Path(base_path)
    # print("base_path", base_path)
    camera_params = {}

    # Exclude fisheye cameras
    exclude_keywords = ["fisheye"]

    # Identify relevant files
    extrinsic_files = [f for f in base_path.glob("*_extrinsic_params.json") if all(k not in f.name for k in exclude_keywords)]
    intrinsic_files = [f for f in base_path.glob("*_intrinsic_params.json") if all(k not in f.name for k in exclude_keywords)]

    # Load extrinsics
    for file in extrinsic_files:
        camera_name = file.stem.replace("_extrinsic_params", "")
        print("camera_name", camera_name)
        with open(file, "r") as f:
            data = json.load(f)
            camera_params[camera_name] = {"extrinsic": data["extrinsic"]}

    # Load intrinsics
    for file in intrinsic_files:
        camera_name = file.stem.replace("_intrinsic_params", "")
        if camera_name in camera_params:  # Ensure extrinsics exist for this camera
            with open(file, "r") as f:
                data = json.load(f)
                camera_params[camera_name]["intrinsic"] = data["intrinsic"]

    return camera_params


def load_depths(root_dir: str, camera_name: str):
    cam_path = Path(root_dir)
    all_imgs = sorted(list(cam_path.glob(f"{camera_name}*")))
    # print("all_imgs", all_imgs)
    return [np.array(Image.open(f)).astype(np.float32) / 1000 for f in all_imgs]

def load_video_frames(video_path, fps=30):
    """
    Load video frames using PyAV directly, ensuring proper decoding of AV1 codec.

    Parameters:
        video_path (str): Path to the video file.
        fps (float): Frames per second of the video (unused in this direct frame decoding).

    Returns:
        list: List of RGB frames extracted from the video.
    """
    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        # Convert PyAV frame to numpy array and ensure RGB format
        frame_rgb = frame.to_image()  # PIL Image
        frames.append(np.array(frame_rgb))
    container.close()
    return frames

def load_local_dataset(episode_id: int, src_path: str, task_id: int) -> Union[list, None]:
    """Load local dataset and return a dict with observations and actions"""

    ob_dir = Path(src_path) / f"observations/{task_id}/{episode_id}"
    depth_imgs = load_depths(ob_dir / "depth", HEAD_DEPTH)
    proprio_dir = Path(src_path) / f"proprio_stats/{task_id}/{episode_id}"

    # print("proprio_dir", proprio_dir)

    with h5py.File(proprio_dir / "proprio_stats.h5") as f:
        state_joint = np.array(f["state/joint/position"])
        state_effector = np.array(f["state/effector/position"])
        state_end_pos = np.array(f["state/end/position"])
        state_head = np.array(f["state/head/position"])
        state_waist = np.array(f["state/waist/position"])
        action_joint = np.array(f["action/joint/position"])
        action_effector = np.array(f["action/effector/position"])
        action_head = np.array(f["action/head/position"])
        action_waist = np.array(f["action/waist/position"])
        action_velocity = np.array(f["action/robot/velocity"])

    # print("state_end_pos", state_end_pos.shape)
    # print("state_effector", state_effector.shape)
    states_value = np.hstack(
        [state_joint, state_effector, state_head, state_waist]
    ).astype(np.float32)

    # print("states_value", states_value.shape)
    assert (
        action_joint.shape[0] == action_effector.shape[0]
    ), f"shape of action_joint:{action_joint.shape};shape of action_effector:{action_effector.shape}"
    action_value = np.hstack(
        [action_joint, action_effector, action_head, action_waist, action_velocity]
    ).astype(np.float32)

    assert len(depth_imgs) == len(
        states_value
    ), f"Number of images and states are not equal"
    assert len(depth_imgs) == len(
        action_value
    ), f"Number of images and actions are not equal"
    frames = [
        {
            "observation.images.cam_top_depth": depth_imgs[i],
            "observation.state": states_value[i],
            "state_position": state_end_pos[i],
            "action": action_value[i],
        }
        for i in range(len(depth_imgs))
    ]

    v_path = ob_dir / "videos"
    videos = {
        "observation.images.top_head": v_path / HEAD_COLOR,
        "observation.images.hand_left": v_path / HAND_LEFT_COLOR,
        "observation.images.hand_right": v_path / HAND_RIGHT_COLOR,
        "observation.images.head_center_fisheye": v_path / HEAD_CENTER_FISHEYE_COLOR,
        "observation.images.head_left_fisheye": v_path / HEAD_LEFT_FISHEYE_COLOR,
        "observation.images.head_right_fisheye": v_path / HEAD_RIGHT_FISHEYE_COLOR,
        "observation.images.back_left_fisheye": v_path / BACK_LEFT_FISHEYE_COLOR,
        "observation.images.back_right_fisheye": v_path / BACK_RIGHT_FISHEYE_COLOR,
    }
    camera_path = Path(src_path) / f"parameter/{task_id}/{episode_id}/camera"
    camera_parameters = load_camera_parameters(camera_path)
    return frames, videos, camera_parameters

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=327)
    parser.add_argument("--src_path", type=str, default="/Users/zubairirshad/Downloads/agibotworld-alpha_tiny")

    args = parser.parse_args()
    # task_id = '327'
    task_id = args.task_id
    src_path = args.src_path
    # src_path = '/Users/zubairirshad/Downloads/agibotworld-alpha_tiny'
    all_subdir = sorted(
        [
            f.as_posix()
            for f in Path(src_path).glob(f"observations/{task_id}/*")
            if f.is_dir()
        ]
    )


    # Get all episode id
    all_subdir_eids = [int(Path(path).name) for path in all_subdir]

    raw_datasets_before_filter = [
        load_local_dataset(subdir, src_path=src_path, task_id=task_id)
        for subdir in tqdm(all_subdir_eids)
    ]

    dataset = raw_datasets_before_filter[0]



    rgb_frames = load_video_frames(dataset[1]["observation.images.top_head"])
    frames = dataset[0]
    camera_parameters = dataset[2]

    print("camera_parameters", camera_parameters)

    intrinsic_params = camera_parameters['head']['intrinsic']

    extrinsic_params = camera_parameters['head']['extrinsic']

    extrinsic_params_left = camera_parameters['hand_left']['extrinsic']
    extrinsic_params_right = camera_parameters['hand_right']['extrinsic']

    rgb_image = rgb_frames[0]

    # Create Open3D intrinsic matrix
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width=rgb_image.shape[1],
        height=rgb_image.shape[0],
        fx=intrinsic_params['fx'],
        fy=intrinsic_params['fy'],
        cx=intrinsic_params['ppx'],
        cy=intrinsic_params['ppy']
    )


    # visualize_pointcloud_with_end_effector(rgb_frames, frames, frames, intrinsic, extrinsic_params)


    # Create extrinsic transformation matrix

    extrinsic_matrix = get_transformation_matrix(extrinsic_params)
    extrinsic_matrix_left = get_transformation_matrix(extrinsic_params_left)
    extrinsic_matrix_right = get_transformation_matrix(extrinsic_params_right)

    frame_idx=0
    end_effector_pos = dataset[0]
    # print("end_effector_posx", end_effector_pos)

    video_name = "rgb_depth"+str(task_id)+".mp4"
    create_rgb_depth_video(rgb_frames, frames, video_name, clip_value=2.0, fps=30)

    rgb_image = rgb_frames[frame_idx]
    depth_image = frames[frame_idx]["observation.images.cam_top_depth"]

    intrinsic_params = camera_parameters['head']['intrinsic']

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb_image), o3d.geometry.Image(depth_image), depth_scale=1.0, depth_trunc=2.0, convert_rgb_to_intensity=False
    )
    all_pcd = []

    num_future=100
    future_spheres = []
    for future_idx in range(1, num_future + 1):  # Future positions
        future_frame_idx = frame_idx + future_idx
        if future_frame_idx < len(end_effector_pos):
            future_left = end_effector_pos[future_frame_idx]["state_position"][0]
            future_right = end_effector_pos[future_frame_idx]["state_position"][1]

            # Create spheres for future positions
            future_left_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            future_left_sphere.translate(future_left)
            future_left_sphere.paint_uniform_color(get_rainbow_color(future_idx, num_future))

            future_right_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            future_right_sphere.translate(future_right)
            future_right_sphere.paint_uniform_color(get_rainbow_color(future_idx, num_future))

            # vis.add_geometry(future_left_sphere)
            # vis.add_geometry(future_right_sphere)

            all_pcd.append(future_left_sphere)
            all_pcd.append(future_right_sphere)

    # Create Open3D intrinsic matrix
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width=rgb_image.shape[1],
        height=rgb_image.shape[0],
        fx=intrinsic_params['fx'],
        fy=intrinsic_params['fy'],
        cx=intrinsic_params['ppx'],
        cy=intrinsic_params['ppy']
    )

    o3d_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    # o3d_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # Flip to align correctly
    o3d_pcd.transform(extrinsic_matrix)  # Apply extrinsic transformation

    FOR_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=np.array([0., 0., 0.]))
    FOR_cam.transform(extrinsic_matrix)
    FOR_cam.paint_uniform_color([1, 0, 0])

    FOR_cam_left = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=np.array([0., 0., 0.]))
    FOR_cam_left.transform(extrinsic_matrix_left)
    FOR_cam_left.paint_uniform_color([0, 1, 0])

    FOR_cam_right = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=np.array([0., 0., 0.]))
    FOR_cam_right.transform(extrinsic_matrix_right)
    FOR_cam_right.paint_uniform_color([0, 0, 1])

    all_pcd.append(FOR_cam_left)
    all_pcd.append(FOR_cam_right)

    all_pcd.append(FOR_cam)

    all_pcd.append(o3d_pcd)

    o3d.visualization.draw_geometries(all_pcd)
