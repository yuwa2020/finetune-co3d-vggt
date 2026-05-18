import os
import json
import logging
import random
import numpy as np

from data.base_dataset import BaseDataset
from data.dataset_util import read_image_cv2, threshold_depth_map


class ScanNetPPDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        SCANNETPP_DIR: str = None,
        len_train: int = 10000,
        len_test: int = 1000,
        min_num_images: int = 10,
    ):
        super().__init__(common_conf=common_conf)

        if SCANNETPP_DIR is None:
            raise ValueError("SCANNETPP_DIR must be specified.")

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.SCANNETPP_DIR = SCANNETPP_DIR

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        raw_dir = os.path.join(SCANNETPP_DIR, "raw")
        try:
            scene_ids = sorted(
                s for s in os.listdir(raw_dir)
                if not s.startswith(".") and os.path.isdir(os.path.join(raw_dir, s))
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"SCANNETPP_DIR/raw not found: {raw_dir}")

        if self.debug:
            scene_ids = scene_ids[:2]

        self.data_store = {}

        for scene_id in scene_ids:
            dslr_dir = os.path.join(raw_dir, scene_id, "dslr")
            images_txt = os.path.join(dslr_dir, "colmap", "images.txt")
            cameras_txt = os.path.join(dslr_dir, "colmap", "cameras.txt")
            split_json = os.path.join(dslr_dir, "train_test_lists.json")
            undistorted_dir = os.path.join(dslr_dir, "undistorted_images")

            if not all(os.path.exists(p) for p in [images_txt, cameras_txt, split_json, undistorted_dir]):
                continue

            try:
                with open(split_json) as f:
                    split_data = json.load(f)
                split_key = "train" if split == "train" else "test"
                valid_files = set(split_data.get(split_key, []))
                if not valid_files:
                    continue

                # OPENCV_FISHEYE: CAMERA_ID MODEL W H fx fy cx cy k1 k2 k3 k4
                # Using undistorted images — treat as pinhole (fx, fy, cx, cy only)
                cameras = {}
                with open(cameras_txt) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split()
                        cam_id = int(parts[0])
                        fx, fy = float(parts[4]), float(parts[5])
                        cx, cy = float(parts[6]), float(parts[7])
                        cameras[cam_id] = (fx, fy, cx, cy)

                # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
                frames = []
                with open(images_txt) as f:
                    lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
                for i in range(0, len(lines), 2):
                    parts = lines[i].split()
                    qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                    cam_id = int(parts[8])
                    name = parts[9]

                    if name not in valid_files or cam_id not in cameras:
                        continue

                    img_path = os.path.join(undistorted_dir, name)
                    if not os.path.exists(img_path):
                        continue

                    fx, fy, cx, cy = cameras[cam_id]
                    intri = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
                    extri = _quat_trans_to_extri(qw, qx, qy, qz, tx, ty, tz)

                    frames.append({"img_path": img_path, "extri": extri, "intri": intri})

                if len(frames) < min_num_images:
                    continue

                self.data_store[scene_id] = frames

            except Exception as e:
                logging.warning(f"ScanNetPP: skipping scene {scene_id}: {e}")

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: ScanNetPP scenes loaded: {self.sequence_list_len}")
        logging.info(f"{status}: ScanNetPP dataset length: {len(self)}")

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        frames = self.data_store[seq_name]

        if ids is None:
            ids = np.random.choice(len(frames), img_per_seq, replace=self.allow_duplicate_img)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks, original_sizes = [], [], [], []

        for idx in ids:
            frame = frames[idx]
            image = read_image_cv2(frame["img_path"])
            for _ in range(5):
                if image is not None:
                    break
                frame = random.choice(frames)
                image = read_image_cv2(frame["img_path"])
            if image is None:
                if images:
                    images.append(images[-1])
                    depths.append(depths[-1])
                    extrinsics.append(extrinsics[-1])
                    intrinsics.append(intrinsics[-1])
                    cam_points.append(cam_points[-1])
                    world_points.append(world_points[-1])
                    point_masks.append(point_masks[-1])
                    original_sizes.append(original_sizes[-1])
                continue

            original_size = np.array(image.shape[:2])
            # Pass zero depth (no depth data available) — zeros = all invalid pixels
            depth_map = np.zeros(image.shape[:2], dtype=np.float32)

            (
                image, depth_map, extri_opencv, intri_opencv,
                world_coords_points, cam_coords_points, point_mask, _,
            ) = self.process_one_image(
                image, depth_map, frame["extri"], frame["intri"],
                original_size, target_image_shape, filepath=frame["img_path"],
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        return {
            "seq_name": "scannetpp_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }


def _quat_trans_to_extri(qw, qx, qy, qz, tx, ty, tz):
    """COLMAP quaternion + translation → 3×4 extrinsic (camera-from-world, OpenCV)."""
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])
    t = np.array([tx, ty, tz])
    return np.hstack([R, t.reshape(3, 1)]).astype(np.float64)
