"""
Generate and save augmented training images to disk.
This pre-computes augmentations to speed up training.
"""
import os
import sys
from pathlib import Path
from PIL import Image
import hashlib
import json
from tqdm import tqdm
from torchvision import transforms

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.transform import IMAGENET_MEAN, IMAGENET_STD


def get_augmentation_only_transforms():
    """
    Get augmentation transforms WITHOUT tensor conversion and normalization.
    This is for saving augmented images to disk (as PIL Images).
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(degrees=10),
        # Note: No ToTensor() or Normalize() - we save as images
    ])


def compute_dataset_hash(source_dir):
    """
    Compute a hash of the source dataset to detect changes.
    Uses file count, file sizes, and modification times.
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        return None
    
    # Collect file info
    file_info = []
    for dog_folder in sorted(source_path.iterdir()):
        if not dog_folder.is_dir():
            continue
        
        images = set()
        images.update(dog_folder.glob('*.jpg'))
        images.update(dog_folder.glob('*.jpeg'))
        images.update(dog_folder.glob('*.JPEG'))
        images.update(dog_folder.glob('*.png'))
        images.update(dog_folder.glob('*.PNG'))
        
        for img_path in sorted(images):
            stat = img_path.stat()
            file_info.append(f"{img_path.name}:{stat.st_size}:{stat.st_mtime}")
    
    # Create hash
    info_string = "\n".join(file_info)
    return hashlib.md5(info_string.encode()).hexdigest()


def generate_augmented_images(
    source_dir,
    output_dir,
    num_augmentations=5,
    force_regenerate=False
):
    """
    Generate augmented images from source dataset.
    
    Args:
        source_dir: Path to original training images (e.g., 'data/train')
        output_dir: Path to save augmented images (e.g., 'data/train_augmented')
        num_augmentations: Number of augmented versions to create per original image
        force_regenerate: If True, regenerate even if dataset hasn't changed
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        raise ValueError(f"Source directory does not exist: {source_dir}")
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset has changed
    hash_file = output_path / '.dataset_hash.json'
    current_hash = compute_dataset_hash(source_dir)
    
    if hash_file.exists() and not force_regenerate:
        with open(hash_file, 'r') as f:
            saved_data = json.load(f)
            saved_hash = saved_data.get('hash')
            saved_num_aug = saved_data.get('num_augmentations', num_augmentations)
        
        if saved_hash == current_hash and saved_num_aug == num_augmentations:
            print(f"✓ Augmented dataset is up-to-date (hash: {current_hash[:8]}...)")
            print(f"  Using existing augmented images in: {output_dir}")
            return True
    
    print(f"Generating {num_augmentations} augmented versions per image...")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    
    # Get augmentation transform (without tensor conversion)
    aug_transform = get_augmentation_only_transforms()
    
    # Process each dog folder
    total_images = 0
    total_augmented = 0
    
    for dog_folder in tqdm(sorted(source_path.iterdir()), desc="Processing dogs"):
        if not dog_folder.is_dir():
            continue
        
        dog_id = dog_folder.name
        output_dog_dir = output_path / dog_id
        output_dog_dir.mkdir(exist_ok=True)
        
        # Find all images
        images = set()
        images.update(dog_folder.glob('*.jpg'))
        images.update(dog_folder.glob('*.jpeg'))
        images.update(dog_folder.glob('*.JPEG'))
        images.update(dog_folder.glob('*.png'))
        images.update(dog_folder.glob('*.PNG'))
        images = sorted(list(images))
        
        for img_path in images:
            total_images += 1
            
            try:
                # Load original image
                img = Image.open(img_path).convert('RGB')
                original_name = img_path.stem  # Name without extension
                extension = img_path.suffix.lower()
                if extension == '.jpeg':
                    extension = '.jpg'
                
                # Generate multiple augmented versions
                for aug_idx in range(num_augmentations):
                    # Apply augmentation (different each time due to randomness)
                    augmented_img = aug_transform(img)
                    
                    # Save augmented image
                    aug_filename = f"{original_name}_aug{aug_idx}{extension}"
                    aug_path = output_dog_dir / aug_filename
                    augmented_img.save(aug_path, quality=95)
                    total_augmented += 1
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    # Save dataset hash
    hash_data = {
        'hash': current_hash,
        'num_augmentations': num_augmentations,
        'source_dir': str(source_path),
        'total_original_images': total_images,
        'total_augmented_images': total_augmented
    }
    
    with open(hash_file, 'w') as f:
        json.dump(hash_data, f, indent=2)
    
    print(f"\n✓ Generated {total_augmented} augmented images from {total_images} original images")
    print(f"  Saved to: {output_dir}")
    print(f"  Dataset hash saved to: {hash_file}")
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate augmented training images')
    parser.add_argument('--source_dir', type=str, default='data/train',
                       help='Source directory with original training images')
    parser.add_argument('--output_dir', type=str, default='data/train_augmented',
                       help='Output directory for augmented images')
    parser.add_argument('--num_augmentations', type=int, default=5,
                       help='Number of augmented versions per original image')
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration even if dataset unchanged')
    
    args = parser.parse_args()
    
    generate_augmented_images(
        args.source_dir,
        args.output_dir,
        num_augmentations=args.num_augmentations,
        force_regenerate=args.force
    )




