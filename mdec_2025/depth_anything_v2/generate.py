import os
import sys
import numpy as np
import torch
from diffusers.pipelines.marigold.marigold_image_processing import MarigoldImageProcessor
from transformers import pipeline, AutoImageProcessor
from tqdm import tqdm
import argparse
import cv2
from PIL import Image
import huggingface_hub as hf

PATH_SELF_DIR = os.path.dirname(__file__)
PATH_MDEC_2025 = os.path.realpath(os.path.join(PATH_SELF_DIR, ".."))
sys.path.append(PATH_MDEC_2025)

from util.syns_patches_accessor import SynsPatchesAccessor

def fourier_features(img):
    # Convert PIL Image to numpy array
    img = np.array(img)
    
    # Process each channel separately
    channels = []
    ffts = []
    
    for channel in cv2.split(img):
        # Normalize channel to [0,1]
        channel = channel.astype(np.float32) / 255.0
        
        # Apply FFT
        fft = np.fft.fft2(channel)
        fft_shift = np.fft.fftshift(fft)
        ffts.append(fft_shift)
        
        # Get magnitude spectrum
        magnitude = np.abs(fft_shift)
        # Apply log transform to compress dynamic range
        log_magnitude = np.log1p(magnitude)
        
        # Normalize to [0,1] range
        log_magnitude = (log_magnitude - np.min(log_magnitude)) / (np.max(log_magnitude) - np.min(log_magnitude))
        channels.append(log_magnitude)
    
    return ffts, np.stack(channels, axis=-1)

def apply_fourier_filter_color(fft_shifts, filter_type="HighEmphasis", cutoff_freq=10, gain_low=5.0, gain_high=1.0):
    """Apply Fourier filter to each color channel separately"""
    filtered_channels = []
    
    for fft_shift in fft_shifts:
        # Create frequency distance grid
        rows, cols = fft_shift.shape
        crow, ccol = rows // 2, cols // 2
        x = np.arange(cols)
        y = np.arange(rows)
        u, v = np.meshgrid(x, y)
        D = np.sqrt((u - ccol)**2 + (v - crow)**2)
        
        # Create filter mask
        if filter_type == "HighEmphasis":
            hpf_base = 1 - np.exp(-D**2 / (2 * cutoff_freq**2))
            H = gain_low + gain_high * hpf_base
        
        # Apply filter
        fft_filtered_shifted = fft_shift * H
        fft_filtered = np.fft.ifftshift(fft_filtered_shifted)
        image_filtered = np.fft.ifft2(fft_filtered)
        image_filtered = np.real(image_filtered)
        
        # Normalize
        image_filtered = (image_filtered - np.min(image_filtered)) / (np.max(image_filtered) - np.min(image_filtered))
        filtered_channels.append(image_filtered)
    
    return np.stack(filtered_channels, axis=-1)

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
    
    # The line `hf.login(token=args.hf_token)` is logging into the Hugging Face Hub using the provided
    # token. The `hf.login()` function is used to authenticate the user with the Hugging Face Hub by
    # providing an API token. This token allows the script to access and interact with models and
    # resources hosted on the Hugging Face Hub.

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
        # Apply Fourier transform preprocessing
        ffts, _ = fourier_features(img)
        img_filtered = apply_fourier_filter_color(ffts, filter_type="HighEmphasis", cutoff_freq=10, gain_low=5.0, gain_high=1.0)
        img_filtered = Image.fromarray((img_filtered * 255).astype(np.uint8))
        
        # Generate depth prediction
        disparity = pipe(img_filtered)["predicted_depth"]
        out.append(np.array(disparity.squeeze()))
        vis = visualize_disparity_as_affine_invariant(disparity)
        vis.save(f"{PATH_OUTPUTS}/{idx:04d}_{img_path.replace('/', '_')}")

    out = np.stack(out)
    np.savez(f"{PATH_SELF_DIR}/pred_{SPLIT}.npz", pred=out, pred_type="disparity")