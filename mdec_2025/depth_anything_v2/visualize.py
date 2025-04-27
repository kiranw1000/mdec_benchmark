import matplotlib.pyplot as plt
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, default='pred_val.npz')
    parser.add_argument('--image_indices', nargs='+', type=int, default=[0, 1, 2])
    args = parser.parse_args()
    
    print(args.image_indices)
    
    num_images = len(args.image_indices)

    # Load the prediction files
    pred_val = np.load(args.pred_path)['pred'] 
    # pred_test = np.load('/Users/kiran/Documents_local/CSE_5524/mdec_benchmark/pred.npz')['pred']

    def visualize_depth(disparity):
        mask_valid = disparity > 0
        depth = 1.0 / np.clip(disparity, 1e-6, None)
        depth_valid = depth[mask_valid]
        d_min = np.quantile(depth_valid, 0.05)
        d_max = np.quantile(depth_valid, 0.95)
        depth = np.clip((depth - d_min) / (d_max - d_min + 1e-6), 0, 1)
        return depth

    # Create figure with subplots
    fig, axes = plt.subplots(num_images, 1)
    fig.suptitle('Validation Set Depth Prediction')

    # Show a few example comparisons
    for i in range(num_images):
        # Val set prediction
        depth_val = visualize_depth(pred_val[args.image_indices[i]])
        axes[i].imshow(depth_val, cmap='plasma')
        axes[i].set_title(f'Validation Set - Image {args.image_indices[i]}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    # Print some statistics
    print("Validation set predictions shape:", pred_val.shape)
    print("\nValidation set stats:")
    print(f"Mean: {pred_val.mean():.3f}")
    print(f"Std: {pred_val.std():.3f}")
    print(f"Min: {pred_val.min():.3f}")
    print(f"Max: {pred_val.max():.3f}")
