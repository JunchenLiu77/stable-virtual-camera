import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
import math
import tyro
from einops import rearrange, repeat
import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from train.runner import Launcher, LauncherConfig, nested_to_device
import imageio.v3 as iio

from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.sampling import (
    DDPMDiscretization,
    DiscreteDenoiser,
    EulerEDMSampler,
    MultiviewCFG,
)
from seva.utils import load_model
from seva.geometry import get_plucker_coordinates
from train.dataset import TrainDataset
from train.runner import set_random_seed


@dataclass
class SEVALauncherConfig(LauncherConfig):
    use_torch_compile: bool = True
    output_dir: str = "work_dirs/dbg"
    max_steps: int = 100_000
    ckpt_every: int = 1000
    print_every: int = 100
    visual_every: int = 100
    lr: float = 4e-4
    warmup_steps: int = 2500
    disable_ae_decoder: bool = False

    # dataset
    data_basedir: str = "./data_processed/dl3dv-10k/"
    latent_basedir: str = "/scratch/one_month/2025_02/junchenliu/seva/dl3dv-10k/"
    batch_scenes: int = 2
    patch_size: int = 256
    total_views: int = 21

    # sampler settings
    video_save_fps: float = 5.0
    cfg: float = 2.0
    seed: int = 23
    steps: int = 50
    s_churn: float = 0.0
    s_tmin: float = 0.0
    s_tmax: float = 999.0
    s_noise: float = 1


class SEVALauncher(Launcher):
    config: SEVALauncherConfig

    def process(
        self,
        data: Dict[str, Any],
        training: bool = True,
        p_drop: float = 0.1,
        scene_scale: float = 2.0,
    ) -> Tuple[torch.Tensor, dict, dict]:
        """
        Before this function, resize and center crop have done and K are updated correspondingly.
        View sampling logic is implemented in the dataset.py.
        This function is responsible for:
        1. determine the input_mask.
        2. normalize camera intrinsics.
        3. perform camera extrinsic normalization.
        4. pack data for conditional generation.
        5. pack data for sampling/visualization.
        """
        if not training:
            # TODO: what to do for inference?
            raise NotImplementedError("Not implemented for inference.")
        assert "latent" in data and "clip" in data, "latent and clip must be provided"
        B, V, H, W, C = data["image"].shape
        assert H == W == self.config.patch_size, "patch size must be equal to 256"

        c2ws = data["camtoworld"]  # [B, V, 4, 4]
        Ks = data["K"]  # [B, V, 3, 3]
        # images = data["image"]  # [B, V, H, W, 3], data range: [0, 255]
        latents = data["latent"]  # [B, V, 4, H//8, W//8]
        clips = data["clip"]  # [B, V, 1024]

        # 1. determine the input_mask
        # TODO: input/target participation logic @hangg.
        num_inputs = 1
        input_masks_ = torch.tensor(
            [1] * num_inputs + [0] * (V - num_inputs),
            device=self.device,
            dtype=torch.bool,
        )  # [V]
        input_masks = input_masks_[None].repeat(B, 1)  # [B, V]

        # 2. normalize camera intrinsics.
        Ks = Ks / Ks.new_tensor([W, H, 1])[:, None]

        # 3. perform camera extrinsic normalization.
        # 3.1 translate camera center to the origin (following @hangg's heuristic).
        positions = c2ws[:, :, :3, 3]
        median_pos = positions.median(0, keepdim=True).values
        dists = torch.norm(positions - median_pos, dim=-1)
        threshold = min(torch.quantile(dists, 0.97) * 10, 1e6)
        valid_masks = dists <= threshold
        valid_pos = torch.zeros_like(positions)
        valid_pos[valid_masks] = positions[valid_masks]
        mean_pos = valid_pos.mean(dim=1, keepdim=True)
        c2ws[:, :, :3, 3] -= mean_pos

        # 3.2 normalize scene scale (following @hangg's heuristic).
        dists = torch.norm(c2ws[:, :, :3, 3], dim=-1)
        normalize_masks = dists[:, 0].abs() > 1e-5  # [B]
        scales = torch.full_like(dists[:, 0], scene_scale)
        scales[normalize_masks] /= dists[:, 0][normalize_masks]
        c2ws[:, :, :3, 3] *= scales[:, None, None]
        w2cs = torch.linalg.inv(c2ws)

        # 4. pack data for conditional generation.
        pluckers = []
        for i in range(B):
            pluckers.append(
                get_plucker_coordinates(
                    extrinsics_src=w2cs[i, 0],
                    extrinsics=w2cs[i],
                    intrinsics=Ks[i].clone(),
                    mode="plucker",
                    rel_zero_translation=True,
                    target_size=(H // 8, W // 8),
                    return_grid_cam=True,
                )[0]
            )
        pluckers = torch.stack(pluckers, dim=0)  # [B, V, 6, H//8, W//8]

        if np.random.rand() > p_drop:
            input_clips = torch.zeros_like(clips)  # [B, V, 1024]
            input_clips[input_masks] = clips[input_masks]
            crossattn = repeat(
                input_clips.mean(dim=1), "b d -> b v 1 d", v=V
            )  # [B, V, 1, 1024]

            latents_and_masks = F.pad(latents, (0, 0, 0, 0, 0, 1), value=1.0)
            replace = torch.zeros_like(latents_and_masks)
            replace[input_masks] = latents_and_masks[
                input_masks
            ]  # [B, V, 5, H//8, W//8]

            concat = torch.cat(
                [
                    repeat(input_masks, "b t -> b t 1 h w", h=H // 8, w=W // 8),
                    pluckers,
                ],
                dim=2,
            )  # [B, V, 7, H//8, W//8]
        else:
            crossattn = torch.zeros(B, V, 1, 1024, device=self.device)
            replace = torch.zeros(B, V, 5, H // 8, W // 8, device=self.device)
            concat = F.pad(pluckers, (0, 0, 0, 0, 1, 0), value=0.0)

        cond = {
            "crossattn": crossattn,  # [B, V, 1, 1024]
            "concat": concat,  # [B, V, 7, H//8, W//8]
            "dense_vector": pluckers,  # [B, V, 6, H//8, W//8]
        }
        if not training:
            cond["replace"] = replace  # [B, V, 5, H//8, W//8]

        # 5. pack first batch data for sampling/visualization.
        value_dict = {}
        # images_ = (images / 255.0) * 2.0 - 1.0  # [B, V, H, W, 3], range: [-1, 1]
        # value_dict["cond_frames"] = rearrange(images_[0], "v h w c -> v c h w")
        value_dict["latents"] = latents[0]
        value_dict["clips"] = clips[0]
        value_dict["cond_frames_mask"] = input_masks[0]
        value_dict["cond_aug"] = 0.0
        value_dict["plucker_coordinate"] = pluckers[0]
        value_dict["c2w"] = c2ws[0]
        value_dict["K"] = Ks[0]
        value_dict["camera_mask"] = input_masks[0]

        return latents, cond, value_dict

    def diffusion_loss(
        self,
        latents: torch.Tensor,  # [B, V, 4, H//8, W//8]
        cond: dict,
        model: SGMWrapper,
        denoiser: DiscreteDenoiser,
        w: torch.Tensor | float = 1.0,
    ):
        """
        hangg's implementation of diffusion loss.
        """
        B, V = latents.shape[:2]
        idx = torch.randint(0, denoiser.num_idx, (B,), device=self.device)
        sigma = denoiser.idx_to_sigma(idx)

        latents = rearrange(latents, "b v c h w -> (b v) c h w")
        sigma = repeat(sigma, "b -> (b v)", v=V)
        sigma_bchw = repeat(sigma, "b -> b 1 1 1")
        noised = latents + torch.randn_like(sigma_bchw) * sigma_bchw
        denoised = denoiser(
            model,
            noised,
            sigma,
            {k: v.flatten(0, 1) for k, v in cond.items()},
            num_frames=V,
        )
        if isinstance(w, torch.Tensor):
            w = w.flatten()
        w = repeat(w * sigma**-2.0, "b -> b 1 1 1")
        loss = (w * F.mse_loss(denoised, latents, reduction="none")).mean()

        del noised, sigma_bchw
        torch.cuda.empty_cache()
        return loss

    @torch.inference_mode()
    def sample(self, state, value_dict):
        """
        Adapted from eval.py, use pre-computed latent and clip instead.
        """
        model = state["model"].to(self.device)
        ae = state["ae"].to(self.device)
        denoiser = state["denoiser"]
        sampler = state["sampler"]
        H = self.version_dict["H"]
        W = self.version_dict["W"]
        C = self.version_dict["C"]
        T = self.version_dict["T"]

        latents = value_dict["latents"]
        clips = value_dict["clips"]
        input_masks = value_dict["cond_frames_mask"]
        pluckers = value_dict["plucker_coordinate"]

        num_samples = [1, self.version_dict["T"]]
        with torch.autocast("cuda"):
            latents = F.pad(latents[input_masks], (0, 0, 0, 0, 0, 1), value=1.0)
            c_crossattn = repeat(clips[input_masks].mean(0), "d -> n 1 d", n=T)

            uc_crossattn = torch.zeros_like(c_crossattn)
            c_replace = latents.new_zeros(T, *latents.shape[1:])
            c_replace[input_masks] = latents
            uc_replace = torch.zeros_like(c_replace)
            c_concat = torch.cat(
                [
                    repeat(
                        input_masks,
                        "n -> n 1 h w",
                        h=pluckers.shape[2],
                        w=pluckers.shape[3],
                    ),
                    pluckers,
                ],
                1,
            )
            uc_concat = torch.cat(
                [pluckers.new_zeros(T, 1, *pluckers.shape[-2:]), pluckers], 1
            )
            c_dense_vector = pluckers
            uc_dense_vector = c_dense_vector
            c = {
                "crossattn": c_crossattn,
                "replace": c_replace,
                "concat": c_concat,
                "dense_vector": c_dense_vector,
            }
            uc = {
                "crossattn": uc_crossattn,
                "replace": uc_replace,
                "concat": uc_concat,
                "dense_vector": uc_dense_vector,
            }

            additional_model_inputs = {"num_frames": T}
            additional_sampler_inputs = {
                "c2w": value_dict["c2w"].to("cuda"),
                "K": value_dict["K"].to("cuda"),
                "input_frame_mask": value_dict["cond_frames_mask"].to("cuda"),
            }
            shape = (math.prod(num_samples), C, H // 8, W // 8)
            randn = torch.randn(shape).to("cuda")

            samples_z = sampler(
                lambda input, sigma, c: denoiser(
                    model,
                    input,
                    sigma,
                    c,
                    **additional_model_inputs,
                ),
                randn,
                scale=self.config.cfg,
                cond=c,
                uc=uc,
                verbose=True,
                **additional_sampler_inputs,
            )
            assert samples_z is not None
            samples = ae.decode(samples_z, chunk_size=1)

        state["ae"] = state["ae"].to("cpu")
        del latents, clips, input_masks, pluckers, samples_z
        torch.cuda.empty_cache()
        return samples

    def train_initialize(self) -> Dict[str, Any]:
        set_random_seed(self.config.seed + self.world_rank)
        # ------------- Setup Args. ------------- #
        self.version_dict = {
            "H": self.config.patch_size,
            "W": self.config.patch_size,
            "T": self.config.total_views,
            "C": 4,
            "f": 8,
        }

        # ------------- Setup Data. ------------- #
        exclude_scenes = set(json.load(open("./assets/dl3dv-10k/bad_scenes.json")))
        scenes = set(os.listdir(self.config.data_basedir))
        scenes = scenes - exclude_scenes
        data_dirs = [os.path.join(self.config.data_basedir, id) for id in scenes]
        latent_dirs = [os.path.join(self.config.latent_basedir, id) for id in scenes]
        trainset = TrainDataset(
            data_dirs=data_dirs,
            patch_size=self.config.patch_size,
            total_views=self.config.total_views,
            latent_dirs=latent_dirs,
        )
        dataloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.config.batch_scenes,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )
        self.logging_on_master(f"Total scenes: {len(trainset)}")

        # ------------- Setup Model. ------------- #
        model = SGMWrapper(load_model(device="cpu", verbose=True))
        ae = AutoEncoder(chunk_size=1).eval()
        # conditioner = CLIPConditioner()
        discretization = DDPMDiscretization()
        denoiser = DiscreteDenoiser(
            discretization=discretization, num_idx=1000, device=self.device
        )
        sampler = EulerEDMSampler(
            discretization=discretization,
            guider=MultiviewCFG(),
            num_steps=self.config.steps,
            s_churn=self.config.s_churn,
            s_tmin=self.config.s_tmin,
            s_tmax=self.config.s_tmax,
            s_noise=self.config.s_noise,
            verbose=True,
            device=self.device,
        )

        # Apply torch.compile for performance optimization if enabled
        if self.config.use_torch_compile:
            model = torch.compile(model)
            ae = torch.compile(ae)
            # conditioner = torch.compile(conditioner)
        print(f"Model is initialized in rank {self.world_rank}")

        # ------------- Setup Optimizer. ------------- #
        params_decay = {
            "params": [p for n, p in model.named_parameters() if "norm" not in n],
            "weight_decay": 0.5,
        }
        params_no_decay = {
            "params": [p for n, p in model.named_parameters() if "norm" in n],
            "weight_decay": 0.0,
        }
        optimizer = torch.optim.AdamW(
            [params_decay, params_no_decay], lr=self.config.lr, betas=(0.9, 0.95)
        )

        # ------------- Setup Scheduler. ------------- #
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.01,
                    total_iters=self.config.warmup_steps,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.config.max_steps - self.config.warmup_steps,
                ),
            ]
        )

        # ------------- Setup Metrics. ------------- #
        psnr_fn = PeakSignalNoiseRatio(data_range=1.0)
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0)
        # Note: careful when comparing with papers: "vgg" or "alex"
        lpips_fn = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        )

        # prepare returns
        state = {
            "model": model.to(self.device),
            "ae": ae,
            # "conditioner": conditioner,
            "denoiser": denoiser,
            "sampler": sampler,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "dataloader": dataloader,
            "dataiter": iter(dataloader),
            "ssim_fn": ssim_fn.to(self.device),
            "psnr_fn": psnr_fn.to(self.device),
            "lpips_fn": lpips_fn.to(self.device),
        }
        print(f"Launcher(train) is intialized in rank {self.world_rank}")

        # debugging
        # data = next(state["dataiter"])
        # data = nested_to_device(data, self.device)
        # latents, cond, value_dict = self.process(data)
        # samples = self.sample(state, value_dict)  # [V, C, H, W]
        # samples = samples.cpu()
        # samples_ = (samples.permute(0, 2, 3, 1) + 1) / 2.0
        # samples_ = (samples_ * 255).clamp(0, 255).to(torch.uint8)
        # iio.imwrite(
        #     (os.path.join(self.visual_dir, f"training.mp4")),
        #     samples_,
        #     fps=self.config.video_save_fps,
        #     macro_block_size=1,
        #     ffmpeg_log_level="error",
        # )
        # exit()
        return state

    def train_iteration(
        self, step: int, state: Dict[str, Any], acc_step: int, *args, **kwargs
    ) -> None:
        dataloader = state["dataloader"]
        dataiter = state["dataiter"]
        model = state["model"]
        denoiser = state["denoiser"]
        model.train()

        try:
            data = next(dataiter)
        except StopIteration:
            dataiter = iter(dataloader)
            data = next(dataiter)
            state["dataiter"] = dataiter
        data = nested_to_device(data, self.device)

        # forward pass
        with torch.amp.autocast("cuda", enabled=self.config.amp, dtype=self.amp_dtype):
            # value_dict is the first batch data for sampling/visualization
            latents, cond, value_dict = self.process(data)
            loss = self.diffusion_loss(latents, cond, model, denoiser)
            del latents, cond
            torch.cuda.empty_cache()

        # save visual (first batch)
        if (
            self.config.visual_every > 0
            and step % self.config.visual_every == 0
            and self.world_rank == 0
            and acc_step == 0
            and not self.config.disable_ae_decoder
        ):
            gt_imgs = (data["image"][0] / 255.0).clamp(0, 1).float()
            gt_imgs = rearrange(gt_imgs, "v h w c -> v c h w")
            samples = self.sample(state, value_dict)  # [V, C, H, W]
            samples = ((samples + 1.0) / 2.0).clamp(0, 1).float()

            samples_ = samples.cpu().permute(0, 2, 3, 1)
            samples_ = (samples_ * 255).to(torch.uint8)
            iio.imwrite(
                (os.path.join(self.visual_dir, "training.mp4")),
                samples_,
                fps=5,
                macro_block_size=1,
                ffmpeg_log_level="error",
            )

        del value_dict
        torch.cuda.empty_cache()

        # calculate metrics (first batch)
        if (
            step % self.config.print_every == 0
            and self.world_rank == 0
            and acc_step == 0
        ):
            if self.config.disable_ae_decoder:
                self.writer.add_scalar("train/loss", loss, step)
                self.logging_on_master(
                    f"Step: {step}, Loss: {loss:.3f} "
                    f"LR: {state['scheduler'].get_last_lr()[0]:.3e}"
                )
            else:
                psnr = state["psnr_fn"](samples, gt_imgs)
                ssim = state["ssim_fn"](samples, gt_imgs)
                lpips = state["lpips_fn"](samples, gt_imgs)
                self.logging_on_master(
                    f"Step: {step}, Loss: {loss:.3f}, PSNR: {psnr:.3f}, "
                    f"SSIM: {ssim:.3f}, LPIPS: {lpips:.3f}, "
                    f"LR: {state['scheduler'].get_last_lr()[0]:.3e}"
                )
                self.writer.add_scalar("train/loss", loss, step)
                self.writer.add_scalar("train/psnr", psnr, step)
                self.writer.add_scalar("train/ssim", ssim, step)
                self.writer.add_scalar("train/lpips", lpips, step)

        return loss

    def test_initialize(
        self,
        model: Optional[torch.nn.Module] = None,
    ) -> Dict[str, Any]:
        pass

    @torch.inference_mode()
    def test_iteration(self, step: int, state: Dict[str, Any]) -> None:
        pass


if __name__ == "__main__":
    cfg = tyro.cli(SEVALauncherConfig)
    launcher = SEVALauncher(cfg)
    launcher.run()
