import os
import sys
import diffusers
import numpy as np
import torch
from diffusers import DDIMScheduler
from tqdm import tqdm

PATH_SELF_DIR = os.path.dirname(__file__)
PATH_MDEC_2025 = os.path.realpath(os.path.join(PATH_SELF_DIR, ".."))
sys.path.append(PATH_MDEC_2025)

from util.syns_patches_accessor import SynsPatchesAccessor


if __name__ == "__main__":
    assert len(sys.argv) == 2 and sys.argv[1] in ("val", "test"), "Usage: python generate.py [val | test]"
    SPLIT = sys.argv[1]
    PATH_SYNS_PATCHES_ZIP = f"{PATH_MDEC_2025}/syns_patches.zip"
    PATH_OUTPUTS = f"{PATH_SELF_DIR}/visualization_{SPLIT}"
    os.makedirs(PATH_OUTPUTS, exist_ok=True)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    pipe = diffusers.MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-depth-v1-0").to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    out = []
    for idx, (img, img_path) in enumerate(tqdm(SynsPatchesAccessor(PATH_SYNS_PATCHES_ZIP, SPLIT), leave=False)):
        depth = pipe(
            img,
            num_inference_steps=10,
            processing_resolution=0,
            ensemble_size=1,
            generator=torch.Generator(device=device).manual_seed(2025),
        ).prediction
        out.append(np.array(depth.squeeze()))
        vis = pipe.image_processor.visualize_depth(depth)[0]
        vis.save(f"{PATH_OUTPUTS}/{idx:04d}_{img_path.replace('/', '_')}")

    out = np.stack(out)
    np.savez(f"{PATH_SELF_DIR}/pred_{SPLIT}.npz", pred=out, pred_type="affine-invariant")
