import os
import shutil
import torch
from skimage import data
from skimage.transform import resize
import matplotlib.pyplot as plt
from vitvqgan.modules.stage1.vit_vqgan import small_config, base_config, ViTVQ
from torch_fidelity import calculate_metrics
from PIL import Image
import numpy as np
from hiq import deterministic


def save_images(images, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, img in enumerate(images):
        img = (img * 255).astype(np.uint8)  # Convert to uint8
        Image.fromarray(img).save(os.path.join(directory, f"image_{i:04d}.png"))


if __name__ == "__main__":
    # Define the configuration using OmegaConf
    config = small_config

    # Initialize the model with the given config
    model = ViTVQ(
        image_key=config.image_key,
        image_size=config.image_size,
        patch_size=config.patch_size,
        encoder=config.encoder,
        decoder=config.decoder,
        quantizer=config.quantizer,
        loss=config.loss,
    )

    # Check if a GPU is available and move the model to the GPU if it is
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load model checkpoint
    checkpoint_path = "/home/fuhwu/workspace/vim/experiments/imagenet_vitvq_small_a100_ine1k/05072024_054557/ckpt/imagenet_vitvq_small_a100_ine1k-epoch=22.ckpt"
    model.init_from_ckpt(checkpoint_path)

    # Set the model to evaluation mode
    model.eval()

    # Load and preprocess the coffee image
    image = data.coffee()
    image_resized = resize(image, (256, 256), anti_aliasing=True)
    input_data = (
        torch.tensor(image_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)
    )

    # Run inference
    with torch.no_grad():
        output, _ = model(input_data)
        output_image = output.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Clip output to [0, 1] range
    output_image = output_image.clip(0, 1)

    # Save input and output images for FID calculation
    save_images([image_resized], "input_images")
    save_images([output_image], "output_images")

    # Calculate FID
    fid_result = calculate_metrics(
        input1="input_images",
        input2="output_images",
        cuda=torch.cuda.is_available(),
        isc=False,
        fid=True,
    )

    fid_score = fid_result["frechet_inception_distance"]
    print(f"FID Score: {fid_score}")

    # Plot the input and output images with FID score
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_resized)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(output_image)
    axes[1].set_title(f"Output Image\nFID Score: {fid_score:.2f}")
    axes[1].axis("off")
    plt.savefig("imgs/small_epoch22.png")
    plt.show()

    # Clean up
    shutil.rmtree("input_images")
    shutil.rmtree("output_images")
