import os
import sys
import numpy as np
import torch
from diffusers.pipelines.marigold.marigold_image_processing import MarigoldImageProcessor
from transformers import pipeline, AutoImageProcessor
from tqdm import tqdm
import argparse

PATH_SELF_DIR = os.path.dirname(__file__)
PATH_MDEC_2025 = os.path.realpath(os.path.join(PATH_SELF_DIR, ".."))
sys.path.append(PATH_MDEC_2025)

from util.syns_patches_accessor import SynsPatchesAccessor


def visualize_disparity_as_affine_invariant(disparity):
    mask_valid = disparity > 0
    depth = 1.0 / disparity.clamp(min=1e-6)
    depth_valid = depth[mask_valid]
    d_min = torch.quantile(depth_valid, 0.05)
    d_max = torch.quantile(depth_valid, 0.95)
    depth = ((depth - d_min) / (d_max - d_min).clamp(min=1e-6)).clamp(0, 1)
    vis = MarigoldImageProcessor.visualize_depth(depth)
    return vis[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--model", type=str, default="5524-Group/1-Epoch-nt")
    args = parser.parse_args()
    SPLIT = args.split
    PATH_SYNS_PATCHES_ZIP = f"{PATH_MDEC_2025}/syns_patches.zip"
    PATH_OUTPUTS = f"{PATH_SELF_DIR}/visualization_{SPLIT}"
    os.makedirs(PATH_OUTPUTS, exist_ok=True)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    pipe = pipeline(
        task="depth-estimation",
        model=args.model,
        image_processor=processor,
        device=device
    )

    out = []
    for idx, (img, img_path) in enumerate(tqdm(SynsPatchesAccessor(PATH_SYNS_PATCHES_ZIP, SPLIT), leave=False)):
        disparity = pipe(img)["predicted_depth"]
        out.append(np.array(disparity.squeeze()))
        vis = visualize_disparity_as_affine_invariant(disparity)
        vis.save(f"{PATH_OUTPUTS}/{idx:04d}_{img_path.replace('/', '_')}")

    out = np.stack(out)
    np.savez(f"{PATH_SELF_DIR}/pred_{SPLIT}.npz", pred=out, pred_type="disparity")