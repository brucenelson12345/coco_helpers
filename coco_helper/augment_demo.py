import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import matplotlib.pyplot as plt

def apply_geometric_transforms(image):
    """
    Apply 7 geometric transforms to the image:
    - 3 rotations (90, 180, 270 degrees)
    - Horizontal flip
    - Vertical flip
    - Rotation + Horizontal flip (90 + hflip)
    - Rotation + Vertical flip (90 + vflip)
    
    Returns a list of 8 images: original + 7 transforms.
    """
    transforms = []
    
    # Original
    transforms.append(('original', image))
    
    # 3 Rotations
    transforms.append(('rotate_90', image.rotate(90, expand=True)))
    transforms.append(('rotate_180', image.rotate(180, expand=True)))
    transforms.append(('rotate_270', image.rotate(270, expand=True)))
    
    # Flips
    transforms.append(('hflip', image.transpose(Image.FLIP_LEFT_RIGHT)))
    transforms.append(('vflip', image.transpose(Image.FLIP_TOP_BOTTOM)))
    
    # Combined: Rotate 90 + flip
    rotated_90 = image.rotate(90, expand=True)
    transforms.append(('rotate90_hflip', rotated_90.transpose(Image.FLIP_LEFT_RIGHT)))
    transforms.append(('rotate90_vflip', rotated_90.transpose(Image.FLIP_TOP_BOTTOM)))
    
    return transforms

def apply_photometric_transforms(image):
    """
    Apply 5 photometric transforms:
    - Blur
    - Saturation adjustment (increase)
    - Grayscale
    - Color jitter (random brightness, contrast, saturation, hue)
    - Noise (additive Gaussian noise)
    
    Returns a list of 5 transformed images.
    """
    transforms = []
    
    # Blur
    blurred = image.filter(ImageFilter.GaussianBlur(radius=2))
    transforms.append(('blur', blurred))
    
    # Saturation (increase by 50%)
    enhancer = ImageEnhance.Color(image)
    saturated = enhancer.enhance(1.5)
    transforms.append(('saturation', saturated))
    
    # Grayscale
    grayscale = image.convert('L').convert('RGB')  # Convert to RGB for consistent shape
    transforms.append(('grayscale', grayscale))
    
    # Color jitter: random brightness, contrast, saturation
    jittered = image.copy()
    # Brightness
    enhancer = ImageEnhance.Brightness(jittered)
    jittered = enhancer.enhance(np.random.uniform(0.8, 1.2))
    # Contrast
    enhancer = ImageEnhance.Contrast(jittered)
    jittered = enhancer.enhance(np.random.uniform(0.8, 1.2))
    # Saturation
    enhancer = ImageEnhance.Color(jittered)
    jittered = enhancer.enhance(np.random.uniform(0.8, 1.2))
    # Hue: no direct PIL support, so skip or use numpy (approximate)
    transforms.append(('color_jitter', jittered))
    
    # Noise
    img_array = np.array(image)
    noise = np.random.normal(0, 15, img_array.shape).astype(np.int16)  # Gaussian noise
    noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    noisy = Image.fromarray(noisy_array)
    transforms.append(('noise', noisy))
    
    return transforms

def augment_image(input_filename, output_dir='augmented_images'):
    """
    Full augmentation pipeline:
    1. Load image
    2. Apply geometric transforms (8 images: original + 7)
    3. Apply 5 photometric transforms on each of the 8 -> 40 images
    4. Save all 48 images (8 geometric + 40 photometric-on-geometric)
    5. Display geometric and photometric transforms
    """
    
    # Load image
    try:
        image = Image.open(input_filename)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Step 1: Geometric Transforms ---
    geometric_images = apply_geometric_transforms(image)
    
    # Save geometric images
    for name, img in geometric_images:
        img.save(os.path.join(output_dir, f"{name}.jpg"))
    
    # --- Step 2: Apply photometric transforms on each geometric image ---
    photometric_count = 0
    for geo_name, geo_img in geometric_images:
        photo_transforms = apply_photometric_transforms(geo_img)
        for photo_name, photo_img in photo_transforms:
            photo_img.save(os.path.join(output_dir, f"{geo_name}_{photo_name}.jpg"))
            photometric_count += 1
    
    print(f"Saved {len(geometric_images)} geometric images.")
    print(f"Saved {photometric_count} photometric-augmented images.")
    print(f"Total images saved: {len(geometric_images) + photometric_count}")
    
    # --- Visualization: Geometric Transforms ---
    fig1 = plt.figure(figsize=(12, 6))
    fig1.suptitle("Geometric Transforms", fontsize=16)
    
    for i, (name, img) in enumerate(geometric_images):
        ax = fig1.add_subplot(2, 4, i + 1)
        ax.imshow(img)
        ax.set_title(name, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # --- Visualization: Photometric Transforms on Original ---
    photo_on_original = apply_photometric_transforms(image)
    
    fig2 = plt.figure(figsize=(15, 6))
    fig2.suptitle("Photometric Transforms (on Original)", fontsize=16)
    
    # Original
    ax = fig2.add_subplot(2, 3, 1)
    ax.imshow(image)
    ax.set_title("original", fontsize=10)
    ax.axis('off')
    
    # Photometric transforms
    for i, (name, img) in enumerate(photo_on_original):
        ax = fig2.add_subplot(2, 3, i + 2)
        ax.imshow(img)
        ax.set_title(name, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Replace with your image path
    input_image_path = "example.jpg"  # Change this to your image file
    augment_image(input_image_path, "augmented_images")