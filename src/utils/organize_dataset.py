"""
Helper script to organize and split dog images into train/val/test folders.
This script helps you organize your dataset correctly.
"""
import os
import shutil
from pathlib import Path
import random
from typing import List, Tuple
import argparse


def split_images(
    images: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split images into train/val/test sets.
    
    Args:
        images: List of image paths
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
        
    Returns:
        (train_images, val_images, test_images)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    random.seed(seed)
    shuffled = images.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_images = shuffled[:n_train]
    val_images = shuffled[n_train:n_train + n_val]
    test_images = shuffled[n_train + n_val:]
    
    return train_images, val_images, test_images


def organize_dataset(
    source_dir: str,
    output_dir: str = "data",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_images_per_dog: int = 2,
    seed: int = 42
):
    """
    Organize dataset from source directory into train/val/test structure.
    
    Expected source structure:
    source_dir/
        dog1/
            image1.jpg
            image2.jpg
            ...
        dog2/
            image1.jpg
            ...
    
    Output structure:
    output_dir/
        train/
            dog1/
                ...
        val/
            dog1/
                ...
        test/
            dog1/
                ...
    
    Args:
        source_dir: Source directory with dog folders
        output_dir: Output directory for organized dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        min_images_per_dog: Minimum images required per dog
        seed: Random seed
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory {source_dir} does not exist!")
        return
    
    # Create output directories
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    test_dir = output_path / "test"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each dog folder
    dog_folders = [d for d in source_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(dog_folders)} dog folders")
    print(f"Split ratio: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}")
    print()
    
    stats = {
        'total_dogs': 0,
        'skipped_dogs': 0,
        'total_images': 0,
        'train_images': 0,
        'val_images': 0,
        'test_images': 0
    }
    
    for dog_folder in sorted(dog_folders):
        dog_id = dog_folder.name
        
        # Get all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        images = []
        for ext in image_extensions:
            images.extend(list(dog_folder.glob(f'*{ext}')))
        
        if len(images) < min_images_per_dog:
            print(f"⚠️  Skipping {dog_id}: Only {len(images)} images (need at least {min_images_per_dog})")
            stats['skipped_dogs'] += 1
            continue
        
        # Split images
        train_imgs, val_imgs, test_imgs = split_images(
            [str(img) for img in images],
            train_ratio,
            val_ratio,
            test_ratio,
            seed
        )
        
        # Ensure at least 1 image in each set
        if len(train_imgs) == 0 or len(val_imgs) == 0 or len(test_imgs) == 0:
            print(f"⚠️  Skipping {dog_id}: Not enough images to split (have {len(images)})")
            stats['skipped_dogs'] += 1
            continue
        
        # Create dog folders in train/val/test
        (train_dir / dog_id).mkdir(exist_ok=True)
        (val_dir / dog_id).mkdir(exist_ok=True)
        (test_dir / dog_id).mkdir(exist_ok=True)
        
        # Copy images
        for img_path in train_imgs:
            src = Path(img_path)
            dst = train_dir / dog_id / src.name
            shutil.copy2(src, dst)
            stats['train_images'] += 1
        
        for img_path in val_imgs:
            src = Path(img_path)
            dst = val_dir / dog_id / src.name
            shutil.copy2(src, dst)
            stats['val_images'] += 1
        
        for img_path in test_imgs:
            src = Path(img_path)
            dst = test_dir / dog_id / src.name
            shutil.copy2(src, dst)
            stats['test_images'] += 1
        
        stats['total_dogs'] += 1
        stats['total_images'] += len(images)
        
        print(f"✅ {dog_id}: {len(images)} images → Train:{len(train_imgs)}, Val:{len(val_imgs)}, Test:{len(test_imgs)}")
    
    # Print summary
    print()
    print("=" * 60)
    print("Organization Complete!")
    print("=" * 60)
    print(f"Total dogs processed: {stats['total_dogs']}")
    print(f"Skipped dogs: {stats['skipped_dogs']}")
    print(f"Total images: {stats['total_images']}")
    print(f"  - Train: {stats['train_images']} ({stats['train_images']/stats['total_images']*100:.1f}%)")
    print(f"  - Val:   {stats['val_images']} ({stats['val_images']/stats['total_images']*100:.1f}%)")
    print(f"  - Test:  {stats['test_images']} ({stats['test_images']/stats['total_images']*100:.1f}%)")
    print()
    print(f"Dataset organized in: {output_path.absolute()}")
    print("=" * 60)


def verify_dataset_structure(data_dir: str):
    """
    Verify that dataset is organized correctly.
    
    Checks:
    1. Same dog IDs in train/val/test
    2. No duplicate images across folders
    3. Both frontal and lateral views present
    """
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"
    
    if not all([train_dir.exists(), val_dir.exists(), test_dir.exists()]):
        print("❌ Missing train/val/test folders!")
        return False
    
    # Get dog IDs from each folder
    train_dogs = set([d.name for d in train_dir.iterdir() if d.is_dir()])
    val_dogs = set([d.name for d in val_dir.iterdir() if d.is_dir()])
    test_dogs = set([d.name for d in test_dir.iterdir() if d.is_dir()])
    
    # Check if same dogs in all folders
    all_dogs = train_dogs | val_dogs | test_dogs
    missing_in_train = all_dogs - train_dogs
    missing_in_val = all_dogs - val_dogs
    missing_in_test = all_dogs - test_dogs
    
    print("=" * 60)
    print("Dataset Verification")
    print("=" * 60)
    
    issues = []
    
    if missing_in_train:
        issues.append(f"❌ Dogs in val/test but not in train: {missing_in_train}")
    if missing_in_val:
        issues.append(f"❌ Dogs in train/test but not in val: {missing_in_val}")
    if missing_in_test:
        issues.append(f"❌ Dogs in train/val but not in test: {missing_in_test}")
    
    if not issues:
        print("✅ Same dog IDs in all folders")
    else:
        for issue in issues:
            print(issue)
    
    # Check for duplicate images
    print("\nChecking for duplicate images...")
    train_images = set()
    val_images = set()
    test_images = set()
    
    for dog_id in train_dogs:
        for img in (train_dir / dog_id).glob("*.jpg"):
            train_images.add(img.name)
        for img in (train_dir / dog_id).glob("*.png"):
            train_images.add(img.name)
    
    for dog_id in val_dogs:
        for img in (val_dir / dog_id).glob("*.jpg"):
            val_images.add(img.name)
        for img in (val_dir / dog_id).glob("*.png"):
            val_images.add(img.name)
    
    for dog_id in test_dogs:
        for img in (test_dir / dog_id).glob("*.jpg"):
            test_images.add(img.name)
        for img in (test_dir / dog_id).glob("*.png"):
            test_images.add(img.name)
    
    train_val_overlap = train_images & val_images
    train_test_overlap = train_images & test_images
    val_test_overlap = val_images & test_images
    
    if train_val_overlap:
        print(f"⚠️  {len(train_val_overlap)} duplicate images between train and val")
    if train_test_overlap:
        print(f"⚠️  {len(train_test_overlap)} duplicate images between train and test")
    if val_test_overlap:
        print(f"⚠️  {len(val_test_overlap)} duplicate images between val and test")
    
    if not (train_val_overlap or train_test_overlap or val_test_overlap):
        print("✅ No duplicate images across folders")
    
    # Check for frontal/lateral views
    print("\nChecking for frontal/lateral views...")
    for dog_id in list(train_dogs)[:5]:  # Check first 5 dogs
        train_imgs = list((train_dir / dog_id).glob("*"))
        has_frontal = any('front' in img.name.lower() or 'frontal' in img.name.lower() for img in train_imgs)
        has_lateral = any('side' in img.name.lower() or 'lateral' in img.name.lower() for img in train_imgs)
        
        if not (has_frontal and has_lateral):
            print(f"⚠️  {dog_id}: Missing frontal or lateral view indicators in filenames")
    
    print("\n" + "=" * 60)
    
    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(description='Organize dog image dataset into train/val/test')
    parser.add_argument('--source_dir', type=str, required=True,
                       help='Source directory with dog folders')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory (default: data)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    parser.add_argument('--min_images', type=int, default=2,
                       help='Minimum images per dog (default: 2)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--verify', type=str, default=None,
                       help='Verify dataset structure (provide data directory path)')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_dataset_structure(args.verify)
    else:
        organize_dataset(
            args.source_dir,
            args.output_dir,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.min_images,
            args.seed
        )
        
        # Verify after organization
        print("\nVerifying organized dataset...")
        verify_dataset_structure(args.output_dir)


if __name__ == '__main__':
    main()

