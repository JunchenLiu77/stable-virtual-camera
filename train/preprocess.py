from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from PIL import Image
import torch
import numpy as np
import imageio.v3 as iio
import tqdm
import os
import glob
from typing import Any, Dict, List
from train.dataset import load_and_maybe_update_meta_info, resize_crop


def is_image_valid(filepath: str) -> bool:
    """Check if the image file is corrupted or not."""
    try:
        if filepath.endswith(".png") or filepath.endswith(".PNG"):
            # Quick Byte Check (Magic Numbers + EOF)
            with open(filepath, "rb") as f:
                start = f.read(8)
                f.seek(-12, os.SEEK_END)
                end = f.read(12)
            return start == b"\x89PNG\r\n\x1a\n" and end.endswith(b"IEND\xaeB\x60\x82")
        elif filepath.endswith(".jpg") or filepath.endswith(".JPG"):
            # Quick Byte Check (Magic Numbers + EOF)
            with open(filepath, "rb") as f:
                start = f.read(2)
                f.seek(-2, os.SEEK_END)
                end = f.read(2)
            return start == b"\xff\xd8" and end == b"\xff\xd9"
        else:
            # Slow Check by loading the pixels
            with Image.open(filepath) as img:
                img.load()
        return True
    except (IOError, ValueError):
        return False


def load_only_frames_from_meta_info(
    data_dir: str,
    meta_info: Dict[str, Any],
    frame_ids: List[int],
    patch_size: int = 256,
) -> Dict[str, Any]:
    frames = meta_info["frames"]

    # Load the images
    images, abs_image_paths = [], []
    for frame_id in frame_ids:
        frame = frames[frame_id]
        rel_image_path = frame["file_path"]
        abs_image_path = os.path.join(data_dir, rel_image_path)
        image = iio.imread(abs_image_path)[..., :3]
        image = resize_crop(image, patch_size)
        images.append(image)
        abs_image_paths.append(abs_image_path)

    return {
        "image": np.stack(images),
        "image_path": abs_image_paths,
    }


class SingleImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dirs: List[str],
        patch_size: int = 256,
    ):
        super().__init__()
        self.data_dirs = np.array(data_dirs).astype(np.string_)
        self.patch_size = patch_size
        indices = []
        for i, data_dir in enumerate(data_dirs):
            valid, meta_info = load_and_maybe_update_meta_info(
                os.path.join(data_dir, "transforms.json")
            )
            assert valid, f"Invalid scene: {data_dir}"
            frames = meta_info["frames"]
            for frame_id in range(len(frames)):
                indices.append((i, frame_id))
        self.indices = np.array(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        scene_id, frame_id = self.indices[item]

        data_dir = str(self.data_dirs[scene_id], encoding="utf-8")
        valid, meta_info = load_and_maybe_update_meta_info(
            os.path.join(data_dir, "transforms.json")
        )
        assert valid, f"Invalid scene: {data_dir}"

        loaded = load_only_frames_from_meta_info(
            data_dir,
            meta_info,
            [frame_id],
            patch_size=self.patch_size,
        )
        loaded = {k: v[0] for k, v in loaded.items()}  # remove the batch dimension

        data = {
            "image": torch.from_numpy(loaded["image"]).float(),
            "image_path": loaded["image_path"],
        }
        return data


def test_ae(
    img_fp: str = "/home/junchenliu/stable-virtual-camera/assets/advance/backyard-7_0.jpg",
    latent_fp: str = "/scratch/one_month/2025_02/junchenliu/seva/dl3dv-140/032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7/images_8/frame_00001.pt",
    use_latent: bool = True,
    save_fp: str = "ae_imgs.png",
):
    device = "cuda:0"
    ae = AutoEncoder().to(device)
    if not use_latent:
        img = Image.open(img_fp)
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).to(device)  # [3, H, W]
        img = img.unsqueeze(0)  # [1, 3, H, W]
        img = img / 255.0 * 2.0 - 1.0
        latent = ae.encode(img, chunk_size=1)
    else:
        latent = torch.load(latent_fp, map_location=device)
    ae_img = ae.decode(latent, chunk_size=1)
    ae_img = (ae_img.permute(0, 2, 3, 1) + 1) / 2.0
    ae_img = (ae_img * 255).clamp(0, 255).to(torch.uint8).cpu()
    iio.imwrite(save_fp, ae_img[0])


@torch.inference_mode()
def process(root_dirs: str, output_dir: str, patch_size: int) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}")
    print("Distributed worker: %d / %d" % (world_rank + 1, world_size))

    os.makedirs(output_dir, exist_ok=True)
    data_dirs = []
    skip_cnt = 0
    for root_dir in root_dirs:
        _data_dirs = sorted(glob.glob(os.path.join(root_dir, "*")))
        for _data_dir in _data_dirs:
            latent_dir = os.path.join(
                output_dir,
                _data_dir.split("/")[-2],
                _data_dir.split("/")[-1],
            )
            images = sorted(glob.glob(os.path.join(_data_dir, "*", "*.png")))
            files = sorted(glob.glob(os.path.join(latent_dir, "*", "*.pt")))
            if len(files) == 2 * len(images):
                if world_rank == 0:
                    print(f"skip {_data_dir}")
                    skip_cnt += 1
                continue
            data_dirs.append(_data_dir)
    data_dirs = data_dirs[world_rank::world_size]
    if world_rank == 0:
        print(f"skip {skip_cnt} scenes")

    ae = AutoEncoder().to(device)
    conditioner = CLIPConditioner().to(device)
    dataset = SingleImageDataset(data_dirs, patch_size=patch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    for data in tqdm.tqdm(dataloader, desc=f"rank {world_rank}"):
        images = data["image"].to(device) / 255.0 * 2.0 - 1.0  # [1, P, P, 3]
        images = images.permute(0, 3, 1, 2)  # [1, 3, P, P]
        file_path = data["image_path"][0]

        latents = ae.encode(images, chunk_size=1)
        latents = latents.detach().cpu()
        clip_conds = conditioner(images)

        # ugly but works
        save_path = os.path.join(
            output_dir,
            file_path.split("/")[-4],
            file_path.split("/")[-3],
            file_path.split("/")[-2],
            file_path.split("/")[-1],
        )
        latents_path = os.path.splitext(save_path)[0] + ".pt"
        clip_path = os.path.splitext(save_path)[0] + "_clip.pt"
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(latents, latents_path)
        torch.save(clip_conds, clip_path)


if __name__ == "__main__":
    """
    Genrate latent and clip condition for dl3dv dataset.

    CUDA_VISIBLE_DEVICES=6,7,8,9 OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=4 -m train.preprocess
    """
    root_dirs = [
        "/home/junchenliu/mvmae/data_processed/dl3dv-10k",
        "/home/junchenliu/mvmae/data_processed/dl3dv-140",
    ]
    output_dir = "/scratch/one_month/2025_02/junchenliu/seva"
    patch_size = 256
    process(root_dirs, output_dir, patch_size)
    # test_ae(use_latent=True)
