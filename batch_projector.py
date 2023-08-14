import os
from tqdm import tqdm
from glob import glob

import click
import numpy as np
import PIL.Image
import torch

import dnnlib
import legacy

from projector import project


# fmt: off
@click.command()
@click.option("--network", "network_pkl", help="Network pickle filename", required=True)
@click.option("--feature-ext", "feature_extractor_pkl", help="Feature extractor model pickle filename", required=True)
@click.option("--target_dir", "target_dirname", help="Directory with target image files to project to", required=True, metavar="DIR")
@click.option("--name", "output_name",    help="Name of the ouput file")
@click.option("--num-steps",              help="Number of optimization steps", type=int, default=1000, show_default=True)
@click.option("--seed",                   help="Random seed", type=int, default=303, show_default=True)
@click.option("--outdir",                 help="Where to save the output images", required=True, metavar="DIR")
# fmt: on
def run_projection(
    network_pkl: str,
    feature_extractor_pkl: str,
    target_dirname: str,
    output_name: str,
    outdir: str,
    seed: int,
    num_steps: int,
):
    """
    Batchwise project given images to the latent space of pretrained network pickle.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print(f"Loading networks from '{network_pkl}'...")
    device = torch.device("cuda")
    with dnnlib.util.open_url(network_pkl) as fp:
        G = (
            legacy.load_network_pkl(fp)["G_ema"].requires_grad_(False).to(device)
        )  # type: ignore

    # Load VGG16 feature detector.
    print(f"Loading feature detection model from '{feature_extractor_pkl}'...")
    with dnnlib.util.open_url(feature_extractor_pkl) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # create directories
    # os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "npz"), exist_ok=True)

    target_fnames = sorted(glob(os.path.join(target_dirname, "*.png")))
    for index, target_fname in enumerate(tqdm(target_fnames)):
        # Load target image.
        target_pil = PIL.Image.open(target_fname).convert("RGB")
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(
            ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)
        )
        target_pil = target_pil.resize(
            (G.img_resolution, G.img_resolution), PIL.Image.LANCZOS
        )
        target_uint8 = np.array(target_pil, dtype=np.uint8)

        # Optimize projection.
        # start_time = perf_counter()
        projected_w_steps = project(
            G,
            target=torch.tensor(
                target_uint8.transpose([2, 0, 1]), device=device
            ),  # pylint: disable=not-callable
            feature_extractor_model=vgg16,
            num_steps=num_steps,
            device=device,
            verbose=False,
        )
        # print(f"Elapsed: {(perf_counter()-start_time):.1f} s")

        filename = output_name if output_name else "proj"
        vector_filename = f"{output_name}_projected_w" if output_name else "projected_w"

        # Save final projected frame and W vector.
        # target_pil.save(os.path.join(outdir, f"{target_filename}.png"))
        projected_w = projected_w_steps[-1]
        synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode="const")
        synth_image = (synth_image + 1) * (255 / 2)
        synth_image = (
            synth_image.permute(0, 2, 3, 1)
            .clamp(0, 255)
            .to(torch.uint8)[0]
            .cpu()
            .numpy()
        )
        index_str = str(index).zfill(len(str(len(target_fnames))))
        PIL.Image.fromarray(synth_image, "RGB").save(
            os.path.join(outdir, f"{filename}_{index_str}.png")
        )
        np.savez(
            os.path.join(outdir, "npz", f"{vector_filename}_{index_str}.npz"),
            w=projected_w.unsqueeze(0).cpu().numpy(),
        )


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
