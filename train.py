import os

import imageio.v3 as iio
import numpy as np
import torch

from seva.eval import do_sample, decode_output
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import (
    DDPMDiscretization,
    DiscreteDenoiser,
    EulerEDMSampler,
    MultiviewCFG,
)
from seva.utils import load_model

VERSION_DICT = {
    "H": 576,
    "W": 576,
    "T": 21,
    "C": 4,
    "f": 8,
    "options": {},
}
options = VERSION_DICT["options"]
options["chunk_strategy"] = "interp"
options["video_save_fps"] = 30.0
options["beta_linear_start"] = 5e-6
options["log_snr_shift"] = 2.4
options["guider_types"] = 1
options["cfg"] = 2.0
options["camera_scale"] = 2.0
options["num_steps"] = 50
options["cfg_min"] = 1.2
options["encoding_t"] = 1
options["decoding_t"] = 1
options["num_inputs"] = None
options["seed"] = 23


@torch.inference_mode()
def save_visual(model, ae, conditioner, denoiser, sampler, value_dict):
    options = VERSION_DICT["options"]
    samples = do_sample(
        model,
        ae,
        conditioner,
        denoiser,
        sampler,
        value_dict,
        H=VERSION_DICT["H"],
        W=VERSION_DICT["W"],
        C=VERSION_DICT["C"],
        F=VERSION_DICT["f"],
        T=VERSION_DICT["T"],
        cfg=3.0,
        **{k: options[k] for k in options if k not in ["cfg", "T"]},
    )
    samples = decode_output(samples, T=VERSION_DICT["T"])

    for sample in samples:
        sample_, media_type = sample.split("/")
        value = samples[sample]
        assert media_type == "image"
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        elif isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        else:
            value = torch.tensor(value)

        value = (value.permute(0, 2, 3, 1) + 1) / 2.0
        value = (value * 255).clamp(0, 255).to(torch.uint8)
        iio.imwrite(
            (
                os.path.join(work_dir, f"{sample_}.mp4")
                if sample_
                else f"{work_dir}.mp4"
            ),
            value,
            fps=5,
            macro_block_size=1,
            ffmpeg_log_level="error",
        )
        os.makedirs(os.path.join(work_dir, sample_), exist_ok=True)
        for i, s in enumerate(value):
            iio.imwrite(
                os.path.join(work_dir, sample_, f"{i:03d}.png"),
                s,
            )


if __name__ == "__main__":
    device = torch.device("cuda")
    work_dir = "work_dirs/tests/"
    os.makedirs(work_dir, exist_ok=True)

    steps = 50
    s_churn = 0.0
    s_tmin = 0.0
    s_tmax = 999.0
    s_noise = 1.0

    model = SGMWrapper(load_model(device="cpu", verbose=True).eval()).to(device)
    ae = AutoEncoder(chunk_size=1).to(device)
    conditioner = CLIPConditioner().to(device)
    discretization = DDPMDiscretization()
    guider = MultiviewCFG()
    denoiser = DiscreteDenoiser(
        discretization=discretization, num_idx=1000, device=device
    )

    sampler = EulerEDMSampler(
        discretization=discretization,
        guider=guider,
        num_steps=steps,
        s_churn=s_churn,
        s_tmin=s_tmin,
        s_tmax=s_tmax,
        s_noise=s_noise,
        verbose=True,
        device=device,
    )
    value_dict = torch.load("assets_demo_cli/value_dict.pth", map_location="cpu")
    save_visual(model, ae, conditioner, denoiser, sampler, value_dict)
