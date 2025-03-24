from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import torch
import numpy as np
import imageio
import json
import cv2


def resize_crop(
    image: np.ndarray, patch_size: int, K: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize and crop the image to have the smallest side equal to `patch_size`."""
    h, w = image.shape[:2]
    if h < w:
        scaled_h = patch_size
        scaled_w = int(w * patch_size / h)
    else:
        scaled_w = patch_size
        scaled_h = int(h * patch_size / w)

    is_downsampling = min(h, w) > patch_size
    interpolation = cv2.INTER_AREA if is_downsampling else cv2.INTER_CUBIC
    image = cv2.resize(image, (scaled_w, scaled_h), interpolation=interpolation)
    if K is not None:
        K = K.copy()
        K[0] *= scaled_w / w
        K[1] *= scaled_h / h

    # Then center-crop the image to patch_size.
    x0 = (scaled_w - patch_size) // 2
    y0 = (scaled_h - patch_size) // 2
    x1 = x0 + patch_size
    y1 = y0 + patch_size
    image = image[y0:y1, x0:x1]
    if K is not None:
        K[0, 2] -= x0
        K[1, 2] -= y0
        return image, K
    else:
        return image


def load_and_maybe_update_meta_info(json_path: str) -> Tuple[bool, Dict]:
    """Load the meta information from the `transforms.json` file.

    If the image paths (e.g., `images/xxx.jpg` ) stored in the `transforms.json` file
    are not found, try with `images_{1, 2, 4, 8}/xxx.jpg` instead, and update the
    camera intrinsics accordingly.
    """
    if not os.path.exists(json_path):
        return False, {}
    with open(json_path, "r") as f:
        meta_info = json.load(f)

    # Check if the image paths are valid
    frames = meta_info["frames"]
    if len(frames) == 0:
        return False, {}

    # Use the jpg version if it exists (for DL3DV)
    if "-jpeg" in json_path:
        for frame in frames:
            frame["file_path"] = os.path.splitext(frame["file_path"])[0] + ".jpeg"

    # Check if the image paths are valid
    maybe_relative_path_to_img = frames[0]["file_path"]
    if maybe_relative_path_to_img.startswith("/"):
        _start_path = os.path.abspath(os.path.dirname(json_path))
        for frame in frames:
            frame["file_path"] = os.path.relpath(frame["file_path"], start=_start_path)
        relative_path_to_img = frames[0]["file_path"]
    else:
        relative_path_to_img = maybe_relative_path_to_img

    abs_path_to_img = os.path.join(os.path.dirname(json_path), relative_path_to_img)
    if not os.path.exists(abs_path_to_img):
        # Try with images_{1, 2, 4, 8}/xxx.jpg
        assert (
            "images/" in relative_path_to_img
        ), f"Invalid image path in the meta info file: {relative_path_to_img}"
        factor = None
        for _factor in [1, 2, 4, 8]:
            _relative_path_to_img = relative_path_to_img.replace(
                "images/", f"images_{_factor}/"
            )
            _abs_path_to_img = os.path.join(
                os.path.dirname(json_path), _relative_path_to_img
            )
            if os.path.exists(_abs_path_to_img):
                factor = _factor
                break

        if factor is None:
            # No valid image path found
            return False, meta_info

        # Found a valid image path, update the meta info
        for frame in frames:
            frame["file_path"] = frame["file_path"].replace(
                "images/", f"images_{factor}/"
            )

        w, h = meta_info["w"], meta_info["h"]
        assert (
            w % factor == 0 and h % factor == 0
        ), f"Invalid factor: {factor} with w={w} and h={h}"

        for key in ["fl_x", "fl_y", "cx", "cy"]:
            meta_info[key] /= factor
        for key in ["w", "h"]:
            meta_info[key] //= factor

    return True, meta_info


def load_frames_from_meta_info(
    data_dir: str,
    meta_info: Dict[str, Any],
    frame_ids: List[int],
    patch_size: int = 256,
    zoom_factor: float = 1.0,
    random_shared_zoom_bounds: Tuple[float, float] = (1.0, 1.0),
    random_zoom: bool = False,
    camera_pose_only: bool = False,
    latent_dir: Optional[str] = None,
) -> Union[Dict[str, Any], np.ndarray]:
    blender2opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )

    # Load the camera intrinsic
    K_raw = np.array(
        [
            [meta_info["fl_x"], 0, meta_info["cx"]],
            [0, meta_info["fl_y"], meta_info["cy"]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    frames = meta_info["frames"]

    # Shortcut for loading only camera poses
    if camera_pose_only:
        c2ws = []
        for frame_id in frame_ids:
            frame = frames[frame_id]
            c2w = np.array(frame["transform_matrix"], dtype=np.float32) @ blender2opencv
            c2ws.append(c2w)
        return np.stack(c2ws)

    # In the case where we use latent with random zoom, the latent is precomputed
    # with the random zoom applied. So we should just load the zoom factor from the
    # disk, that cooresponds to the random zoom factor used during the latent computation.
    precomputed_zoom_factors = None
    if random_zoom and latent_dir is not None:
        fp = os.path.join(latent_dir, "zoom_factor.txt")
        assert os.path.exists(fp), f"Zoom factor file not found: {fp}"
        precomputed_zoom_factors = np.loadtxt(fp)
        assert len(precomputed_zoom_factors) == len(frames)

    # Load the images
    images, Ks, c2ws, abs_image_paths, latents, clips = [], [], [], [], [], []

    for frame_id in frame_ids:
        frame = frames[frame_id]

        rel_image_path = frame["file_path"]
        abs_image_path = os.path.join(data_dir, rel_image_path)
        image = imageio.imread(abs_image_path)[..., :3]

        image, K = resize_crop(image, patch_size, K_raw)

        c2w = np.array(frame["transform_matrix"], dtype=np.float32) @ blender2opencv

        images.append(image)
        Ks.append(K)
        c2ws.append(c2w)
        abs_image_paths.append(abs_image_path)

        if latent_dir is not None:
            abs_latent_dir = os.path.join(
                latent_dir,
                "images_8",
            )
            abs_latent_path = os.path.join(
                abs_latent_dir,
                os.path.basename(os.path.splitext(abs_image_path)[0]) + ".pt",
            )
            abs_clip_path = os.path.join(
                abs_latent_dir,
                os.path.basename(os.path.splitext(abs_image_path)[0]) + "_clip.pt",
            )
            assert os.path.exists(
                abs_latent_path
            ), f"Latent path not found: {abs_latent_path}"
            assert os.path.exists(
                abs_clip_path
            ), f"Clip path not found: {abs_clip_path}"
            latents.append(
                torch.load(abs_latent_path, weights_only=True, map_location="cpu")
            )
            clips.append(
                torch.load(abs_clip_path, weights_only=True, map_location="cpu")
            )

    return {
        "image": np.stack(images),
        "K": np.stack(Ks),
        "camtoworld": np.stack(c2ws),
        "image_path": abs_image_paths,
        "latent": torch.cat(latents, dim=0) if latents else None,
        "clip": torch.cat(clips, dim=0) if latents else None,
    }


class TrainDataset(Dataset):
    def __init__(
        self,
        data_dirs: List[str],
        patch_size: int = 256,
        total_views: int = 21,
        latent_dirs: Optional[List[str]] = None,
    ):
        super().__init__()

        # No list/dict in the dataset, which would cause "memory leak"
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-715050814
        self.data_dirs = np.array(data_dirs).astype(np.bytes_)
        self.patch_size = patch_size
        self.total_views = total_views
        self.latent_dirs = (
            np.array(latent_dirs).astype(np.bytes_) if latent_dirs else None
        )

    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        data_dir = str(self.data_dirs[item], encoding="utf-8")
        latent_dir = (
            str(self.latent_dirs[item], encoding="utf-8")
            if self.latent_dirs is not None
            else None
        )
        valid, meta_info = load_and_maybe_update_meta_info(
            os.path.join(data_dir, "transforms.json")
        )
        assert valid, f"Invalid scene: {data_dir}"

        frames = meta_info["frames"]
        assert (
            len(frames) >= self.total_views
        ), f"Scene {data_dir} has less than {self.total_views} frames."

        # TODO: view sampling logic @hangg.
        # frame_ids = torch.randperm(len(frames))[: self.total_views]
        start_idx = torch.randint(0, len(frames) - self.total_views, (1,)).item()
        frame_ids = torch.arange(start_idx, start_idx + self.total_views)
        frame_ids = frame_ids.tolist()

        loaded = load_frames_from_meta_info(
            data_dir,
            meta_info,
            frame_ids,
            patch_size=self.patch_size,
            latent_dir=latent_dir,
        )

        data = {
            "camtoworld": torch.from_numpy(loaded["camtoworld"]).float(),
            "K": torch.from_numpy(loaded["K"]).float(),
            "image": torch.from_numpy(loaded["image"]).float(),
        }
        if self.latent_dirs is not None:
            data["latent"] = loaded["latent"]
            data["clip"] = loaded["clip"]
        return data
