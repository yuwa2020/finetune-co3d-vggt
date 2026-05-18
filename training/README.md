# Training

This is a re-implementation of the VGGT training framework. This document covers environment setup, dataset preparation, running training, and reproducing our multi-dataset results.

---

## 1. Prerequisites

1. **Install VGGT as a package:**
   ```bash
   pip install -e .
   ```

2. **Prepare datasets and annotations** (see Section 3 for dataset-specific instructions).

---

## 2. Configuration

Configure paths in `training/config/default.yaml` (for Co3D-only) or use the example multi-dataset config at `training/config/multidataset_example.yaml`.

```yaml
checkpoint:
  resume_checkpoint_path: /YOUR/PATH/TO/VGGT_CHECKPOINT
```

---

## 3. Supported Datasets

### Co3D (built-in)
- Download from [official repository](https://github.com/facebookresearch/co3d)
- Download annotation files from [Hugging Face](https://huggingface.co/datasets/JianyuanWang/co3d_anno/tree/main)

```yaml
- _target_: data.datasets.co3d.Co3dDataset
  split: train
  CO3D_DIR: /YOUR/PATH/TO/CO3D
  CO3D_ANNOTATION_DIR: /YOUR/PATH/TO/CO3D_ANNOTATION
  len_train: 10000
```

### VKitti (built-in)
- Download using the provided script `data/preprocess/vkitti.sh`

```yaml
- _target_: data.datasets.vkitti.VKittiDataset
  split: train
  VKitti_DIR: /YOUR/PATH/TO/VKitti
  len_train: 10000
```

### ScanNet++ (new)
- Download from [ScanNet++ official page](https://cy94.github.io/scannetpp/)
- Expected structure:
  ```
  SCANNETPP_DIR/
    raw/
      {scene_id}/
        dslr/
          undistorted_images/   ← uses these (already rectified)
          colmap/
            cameras.txt
            images.txt
          train_test_lists.json
  ```
- Uses `undistorted_images/` to avoid fisheye distortion handling
- Camera intrinsics from `cameras.txt` (OPENCV_FISHEYE model; fx/fy/cx/cy used, distortion ignored)
- Poses from `images.txt` (COLMAP quaternion format)
- Train/test split from `train_test_lists.json`

```yaml
- _target_: data.datasets.scannetpp.ScanNetPPDataset
  split: train
  SCANNETPP_DIR: /YOUR/PATH/TO/SCANNETPP
  len_train: 10000
```

### MegaDepth (new)
- Download from [MegaDepth official page](https://www.cs.cornell.edu/projects/megadepth/)
- Expected structure:
  ```
  MEGADEPTH_DIR/
    megadepth/
      {scene_id}/
        dense0/
          imgs/        ← JPEG images
          depths/      ← HDF5 depth files (.h5, key "depth")
        sparse/
          manhattan/
            0/
              cameras.txt   ← SIMPLE_RADIAL model
              images.txt
  ```
- Scenes are split 90/10 (train/test) by sorted order
- Depth is loaded from `.h5` files when `load_depth: True`
- A pickle cache is written on first load to speed up subsequent runs (saved alongside `MEGADEPTH_DIR/`)

```yaml
- _target_: data.datasets.megadepth.MegaDepthDataset
  split: train
  MEGADEPTH_DIR: /YOUR/PATH/TO/MEGADEPTH
  len_train: 10000
```

---

## 4. Running Training

### Single-dataset (Co3D only)
```bash
torchrun --nproc_per_node=4 launch.py
```
Uses `training/config/default.yaml` with the aggregator frozen.

### Multi-dataset (Co3D + ScanNet++ + MegaDepth)

Copy and edit the example config:
```bash
cp training/config/multidataset_example.yaml training/config/multidataset.yaml
# Edit paths in multidataset.yaml
```

Then run with a custom config:
```bash
torchrun --nproc_per_node=4 launch.py --config-name multidataset
```

Or pass paths at the command line via Hydra overrides:
```bash
torchrun --nproc_per_node=4 launch.py \
  data.train.dataset.dataset_configs.0.CO3D_DIR=/path/to/co3d \
  data.train.dataset.dataset_configs.1.SCANNETPP_DIR=/path/to/scannetpp \
  data.train.dataset.dataset_configs.2.MEGADEPTH_DIR=/path/to/megadepth
```

---

## 5. Experimental Results

All experiments use Co3D fast eval (10 sequences/category × 41 categories) from the [evaluation branch](https://github.com/facebookresearch/vggt/tree/evaluation).

| Model | Frozen | Datasets | AUC@30 | AUC@15 | AUC@5 | AUC@3 |
|---|---|---|---|---|---|---|
| Original VGGT-1B (paper) | — | — | 89.98 | 83.89 | 67.45 | 56.65 |
| Original VGGT-1B (our eval) | — | — | 89.76 | 83.82 | 67.99 | 57.43 |
| Run 2: fine-tune, 50 epochs | aggregator | Co3D 10K | 89.36 | 83.07 | 66.27 | 55.14 |
| Run 3: full model, 50 epochs | nothing | Co3D 10K | 84.67 | 75.22 | 50.19 | 35.14 |
| Run 4: full model, in progress | nothing | Co3D + ScanNet++ + MegaDepth | TBD | TBD | TBD | TBD |

**Key takeaway:** Training all parameters on a single small dataset (Run 3) causes the aggregator to drift from its large-scale pretraining, degrading performance. The multi-dataset mixture (Run 4) is the correct approach — it matches what the original paper did.

### Hardware
Experiments run on 4× NVIDIA L40S (46 GB each). With `max_img_per_gpu=4`, memory usage is ~23-31 GB/GPU depending on whether the aggregator is frozen.

### Config differences vs. original paper

| Setting | Paper | Our runs |
|---|---|---|
| img_size | 518 | 518 ✅ |
| max_img_per_gpu | 48 | 4 (GPU memory limit) |
| max_epochs | 20 | 50 (compensates for smaller batch) |
| LR schedule | warmup 5% + cosine | same ✅ |
| Peak LR | 5e-5 | 5e-5 ✅ |

---

## 6. Common Questions

### Memory management

If you get OOM errors, reduce `max_img_per_gpu` in `default.yaml`. With the aggregator frozen, 518px images use ~14 GB/GPU. With the full model unfrozen, expect ~23-31 GB/GPU.

### Dataset sampling ratio

The ratio between datasets is controlled by `len_train`. For example, to sample Co3D twice as often as ScanNet++:
```yaml
Co3dDataset:    len_train: 10000
ScanNetPPDataset: len_train: 5000
```

### ScanNet++ depth

ScanNet++ DSLR images do not come with per-image depth maps in the standard download. The loader passes zero depth (all invalid) so only the camera loss applies for these images.

### MegaDepth missing images

MegaDepth depth maps and images may be partially missing depending on your download. The loader retries up to 5 times with random frames from the same scene, then duplicates the last valid frame to maintain batch shape consistency.

### Learning rate

LR is sensitive to effective batch size (`max_img_per_gpu × num_gpus × img_per_sample`). If you change the number of GPUs significantly, scale the LR accordingly. Values that worked for us: `5e-5` on 4× L40S.

### Dataloader validation

To visualize dataloader output as a point cloud:

```python
import open3d as o3d
import numpy as np
import torch

def save_ply(points, colors, filename):
    pts = points.reshape(-1, 3).cpu().numpy() if torch.is_tensor(points) else points.reshape(-1, 3)
    rgb = colors.reshape(-1, 3).cpu().numpy() if torch.is_tensor(colors) else colors.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)

save_ply(batch["world_points"][0].reshape(-1, 3),
         batch["images"][0].permute(0, 2, 3, 1).reshape(-1, 3),
         "debug.ply")
```

### Expected coordinate system

Camera poses follow OpenCV `camera-from-world` convention. Depth maps should be aligned with their corresponding camera poses.

### Handling unordered sequences

For unordered sequences, see [Issue #82](https://github.com/facebookresearch/vggt/issues/82) for how to compute frame similarity rankings.
