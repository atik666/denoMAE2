import os
import shutil
import random

source = "/mnt/d/OneDrive - Rowan University/RA/Summer 24/MMAE_Wireless/DenoMAE/data/labeled/"
dest = "/mnt/d/OneDrive - Rowan University/RA/Summer 24/MMAE_Wireless/DenoMAE/data/labeled_mix/"

# Create destination directories if they don't exist
os.makedirs(os.path.join(dest, 'train'), exist_ok=True)
os.makedirs(os.path.join(dest, 'test'), exist_ok=True)

# Constants
SAMPLES_PER_CLASS_TRAIN = 100  # Total 1000 training samples
SAMPLES_PER_CLASS_TEST = 10    # Total 100 test samples

# Get all folders dynamically
folders = [d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d))]
print("Available folders:", folders)

# Get class names from first folder's train directory
first_folder = folders[0]
class_names = [d for d in os.listdir(os.path.join(source, first_folder, 'train')) 
               if os.path.isdir(os.path.join(source, first_folder, 'train', d))]
print("Available classes:", class_names)

def collect_and_copy_images(split_type, samples_per_class):
    for class_name in class_names:
        # Collect all images for this class across folders
        class_images = []
        for folder in folders:
            class_path = os.path.join(source, folder, split_type, class_name)
            if os.path.exists(class_path):
                images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
                class_images.extend(images)
        
        print(f"Class {class_name} - Available images: {len(class_images)}")
        
        # Adjust sample size if needed
        actual_samples = min(samples_per_class, len(class_images))
        if actual_samples < samples_per_class:
            print(f"Warning: Class {class_name} only has {actual_samples} images available")
            
        if actual_samples == 0:
            print(f"Error: No images found for class {class_name}")
            continue
            
        # Randomly select required number of images
        selected_images = random.sample(class_images, actual_samples)
        
        # Copy selected images to destination
        dest_class_path = os.path.join(dest, split_type, class_name)
        os.makedirs(dest_class_path, exist_ok=True)
        
        for img_path in selected_images:
            img_name = os.path.basename(img_path)
            shutil.copy2(img_path, os.path.join(dest_class_path, img_name))
        
        print(f"Copied {actual_samples} images for class {class_name}")

# Process training data
collect_and_copy_images('train', SAMPLES_PER_CLASS_TRAIN)

# Process test data
collect_and_copy_images('test', SAMPLES_PER_CLASS_TEST)

print("Image mixing and copying completed successfully!")