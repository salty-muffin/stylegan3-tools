# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Generate images using pretrained network pickle."""

from typing import List, Optional, Union
import os
import glob

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy


# ----------------------------------------------------------------------------


def parse_paths(s: Union[str, List]) -> List[int]:
    if isinstance(s, list):
        return s
    filenames = []
    if os.path.isdir(s):
        filenames = glob.glob(os.path.join(s, "*.npz"))
        filenames.sort()
    else:
        for p in s.split(","):
            filenames.append(p)
    return filenames


# ----------------------------------------------------------------------------


# fmt: off
@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', required=True)
@click.option('--ws', 'projected_ws',      type=parse_paths, help='One or more projected_w filenames to generate from, or a directory containing .npz files')
@click.option('--noise-mode',              help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir',                  help='Where to save the output images', type=str, required=True, metavar='DIR')
# fmt: on
def generate_images(
    network_pkl: str,
    projected_ws: Optional[str],
    noise_mode: str,
    outdir: str,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    \b
    # Generate uncurated images with truncation using the MetFaces-U dataset
    python gen_images.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl
    """

    # checking for cuda
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        print("cuda is available.")
        device = torch.device("cuda")
    else:
        print("cuda is not available.")
        device = torch.device("cpu")
    print(f'device: "{device}"')

    print(f'Loading networks from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Generate from projected w(s).
    for idx, file in enumerate(projected_ws):
        print(f"Generating image for file {file} ({idx}/{len(projected_ws)}) ...")
        w = torch.tensor(np.load(file)["w"], device=device)
        img = G.synthesis(w, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), "RGB").save(
            os.path.join(outdir, f"{os.path.splitext(os.path.basename(file))[0]}.png")
        )


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
