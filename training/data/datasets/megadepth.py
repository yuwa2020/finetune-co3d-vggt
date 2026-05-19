import os
import logging
import pickle
import random
import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from data.base_dataset import BaseDataset
from data.dataset_util import read_image_cv2, threshold_depth_map


class MegaDepthDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        MEGADEPTH_DIR: str = None,
        len_train: int = 10000,
        len_test: int = 1000,
        min_num_images: int = 10,
        train_ratio: float = 0.9,
    ):
        super().__init__(common_conf=common_conf)

        if MEGADEPTH_DIR is None:
            raise ValueError("MEGADEPTH_DIR must be specified.")

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.MEGADEPTH_DIR = MEGADEPTH_DIR

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        scenes_dir = os.path.join(MEGADEPTH_DIR, "megadepth")
        try:
            all_scene_ids = sorted(
                s for s in os.listdir(scenes_dir)
                if not s.startswith(".") and os.path.isdir(os.path.join(scenes_dir, s))
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"MEGADEPTH_DIR/megadepth not found: {scenes_dir}")

        split_idx = int(len(all_scene_ids) * train_ratio)
        scene_ids = all_scene_ids[:split_idx] if split == "train" else all_scene_ids[split_idx:]

        if self.debug:
            scene_ids = scene_ids[:2]

        # Cache parsed data store to avoid slow NFS re-scan on each run
        cache_key = f"megadepth_{split}_{train_ratio}"
        cache_path = os.path.join(MEGADEPTH_DIR, f".cache_{cache_key}.pkl")

        if not self.debug and os.path.exists(cache_path):
            logging.info(f"MegaDepth: loading cache from {cache_path}")
            with open(cache_path, "rb") as f:
                self.data_store = pickle.load(f)
        else:
            self.data_store = {}
            self._build_data_store(scene_ids, scenes_dir, min_num_images)
            if not self.debug:
                try:
                    with open(cache_path, "wb") as f:
                        pickle.dump(self.data_store, f)
                    logging.info(f"MegaDepth: saved cache to {cache_path}")
                except Exception as e:
                    logging.warning(f"MegaDepth: could not save cache: {e}")

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: MegaDepth scenes loaded: {self.sequence_list_len}")
        logging.info(f"{status}: MegaDepth dataset length: {len(self)}")

    def _build_data_store(self, scene_ids, scenes_dir, min_num_images):
        for scene_id in scene_ids:
            scene_dir = os.path.join(scenes_dir, scene_id)
            imgs_dir = os.path.join(scene_dir, "dense0", "imgs")
            depths_dir = os.path.join(scene_dir, "dense0", "depths")
            manhattan_dir = os.path.join(scene_dir, "sparse", "manhattan")

            if not os.path.isdir(imgs_dir) or not os.path.isdir(manhattan_dir):
                continue

            # Use first available COLMAP reconstruction
            recon_dir = None
            try:
                for sub in sorted(os.listdir(manhattan_dir)):
                    candidate = os.path.join(manhattan_dir, sub)
                    if os.path.isfile(os.path.join(candidate, "cameras.txt")):
                        recon_dir = candidate
                        break
            except Exception:
                continue

            if recon_dir is None:
                continue

            has_depths = os.path.isdir(depths_dir)

            try:
                cameras_txt = os.path.join(recon_dir, "cameras.txt")
                images_txt = os.path.join(recon_dir, "images.txt")

                cameras = {}
                with open(cameras_txt) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split()
                        cam_id = int(parts[0])
                        model = parts[1]
                        if model == "SIMPLE_RADIAL":
                            f_val = float(parts[4])
                            cx, cy = float(parts[5]), float(parts[6])
                            cameras[cam_id] = (f_val, f_val, cx, cy)
                        elif model == "PINHOLE":
                            fx, fy = float(parts[4]), float(parts[5])
                            cx, cy = float(parts[6]), float(parts[7])
                            cameras[cam_id] = (fx, fy, cx, cy)
                        elif model in ("RADIAL", "SIMPLE_PINHOLE"):
                            f_val = float(parts[4])
                            cx, cy = float(parts[5]), float(parts[6])
                            cameras[cam_id] = (f_val, f_val, cx, cy)

                # Parse images — skip per-image existence check to avoid NFS overhead
                frames = []
                with open(images_txt) as f:
                    lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
                for i in range(0, len(lines), 2):
                    parts = lines[i].split()
                    qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                    cam_id = int(parts[8])
                    name = parts[9]

                    if cam_id not in cameras:
                        continue

                    img_path = os.path.join(imgs_dir, name)
                    stem = os.path.splitext(name)[0]
                    depth_path = os.path.join(depths_dir, stem + ".h5") if has_depths else None

                    fx, fy, cx, cy = cameras[cam_id]
                    intri = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
                    extri = _quat_trans_to_extri(qw, qx, qy, qz, tx, ty, tz)

                    frames.append({
                        "img_path": img_path,
                        "depth_path": depth_path,
                        "extri": extri,
                        "intri": intri,
                    })

                if len(frames) < min_num_images:
                    continue

                self.data_store[scene_id] = frames

            except Exception as e:
                logging.warning(f"MegaDepth: skipping scene {scene_id}: {e}")

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        # Retry loop: if a scene has all files missing on NFS, pick a different one
        for _attempt in range(10):
            if self.inside_random or _attempt > 0:
                seq_index = random.randint(0, self.sequence_list_len - 1)

            _seq_name = (seq_name if seq_name is not None and _attempt == 0
                         else self.sequence_list[seq_index])
            frames = self.data_store[_seq_name]

            _ids = (ids if ids is not None and _attempt == 0
                    else np.random.choice(len(frames), img_per_seq, replace=self.allow_duplicate_img))

            target_image_shape = self.get_target_shape(aspect_ratio)

            images, depths, extrinsics, intrinsics = [], [], [], []
            cam_points, world_points, point_masks, original_sizes = [], [], [], []

            for idx in _ids:
                frame = frames[idx]
                image = read_image_cv2(frame["img_path"])
                for _ in range(5):
                    if image is not None:
                        break
                    frame = random.choice(frames)
                    image = read_image_cv2(frame["img_path"])
                if image is None:
                    continue  # padded to len(_ids) after loop

                depth_map = np.zeros(image.shape[:2], dtype=np.float32)
                if self.load_depth and HAS_H5PY and frame["depth_path"] is not None:
                    try:
                        with h5py.File(frame["depth_path"], "r") as hf:
                            depth_map = hf["depth"][:].astype(np.float32)
                        depth_map[~np.isfinite(depth_map)] = 0.0
                        depth_map = threshold_depth_map(depth_map, min_percentile=-1, max_percentile=98)
                        if depth_map is None:
                            depth_map = np.zeros(image.shape[:2], dtype=np.float32)
                    except Exception:
                        depth_map = np.zeros(image.shape[:2], dtype=np.float32)

                original_size = np.array(image.shape[:2])

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

            # Pad to len(_ids) with last valid frame so all batches have the same shape
            while 0 < len(images) < len(_ids):
                images.append(images[-1])
                depths.append(depths[-1])
                extrinsics.append(extrinsics[-1])
                intrinsics.append(intrinsics[-1])
                cam_points.append(cam_points[-1])
                world_points.append(world_points[-1])
                point_masks.append(point_masks[-1])
                original_sizes.append(original_sizes[-1])

            if len(images) > 0:
                return {
                    "seq_name": "megadepth_" + _seq_name,
                    "ids": _ids,
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

            logging.warning(
                f"MegaDepth: all frames missing in scene {_seq_name}, "
                f"retrying with new scene (attempt {_attempt + 1}/10)"
            )

        raise RuntimeError("MegaDepth: failed to load any valid frames after 10 scene attempts")


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
