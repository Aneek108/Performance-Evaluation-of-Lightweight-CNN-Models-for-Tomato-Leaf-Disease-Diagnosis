import os
import shutil
from tqdm import tqdm
import numpy as np
import cv2
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

class _Datasets:

    def __init__(self):
        self.data_path: str = None
        self.class_names: list[str] = None
        self.class_weights: np.ndarray = None
        self.train_metadata: dict = None
        self.val_metadata: dict = None
        self.test_metadata: dict = None
        
    def __train_val_test_split(self, val_size: float, test_size: float, seed: int) -> None:

        # Step 1: Define Dataset Path
        data_dir = "E:/Research & Projects/Tomato Disease Detection/Saved Dataset"  # Your dataset directory
        original_dir = f"{data_dir}/Original"
        splited_dir = f"{data_dir}/Splited"
        dst_dir = f"{splited_dir}/{(1-(val_size + test_size))*100:.0f}_{val_size*100:.0f}_{test_size*100:.0f}"
        self.data_path = dst_dir

        os.makedirs(splited_dir, exist_ok=True)
        os.makedirs(dst_dir, exist_ok=True)
        if len(os.listdir(dst_dir)) != 0:
            raise FileExistsError(f"Folders already exists within {dst_dir}")

        # Step 2: Extract Image Paths & Labels
        self.class_names = sorted(os.listdir(original_dir))  # List of class names
        image_paths, labels = [], []

        for class_idx, class_name in enumerate(self.class_names):
            class_folder = f"{original_dir}/{class_name}"
            for img_name in os.listdir(class_folder):
                image_paths.append(f"{class_folder}/{img_name}")
                labels.append(class_idx)

        # Convert lists to NumPy arrays
        image_paths = np.array(image_paths)
        labels = np.array(labels)
        self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)

        # Step 3: Stratified Train-Test Split (70% Train, 30% Temp)
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, test_size=val_size + test_size, stratify=labels, random_state=seed
        )

        # Step 4: Stratified Validation-Test Split (33% of Temp → Test, 67% of Temp → Val)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=test_size / (val_size + test_size), stratify=temp_labels, random_state=seed
        )

        # Step 5: Print Dataset Details
        self.train_metadata = {
            'path': f"{dst_dir}/train",
            'split_size': 1-(val_size + test_size),
            'n_classes': len(np.unique(train_labels)),
            'n_samples': len(train_labels),
            'class_proportions': np.bincount(train_labels)/len(train_labels),
        }

        self.val_metadata = {
            'path': f"{dst_dir}/val",
            'split_size': val_size,
            'n_classes': len(np.unique(val_labels)),
            'n_samples': len(val_labels),
            'class_proportions': np.bincount(val_labels)/len(val_labels),
        }

        self.test_metadata = {
            'path': f"{dst_dir}/test",
            'split_size': test_size,
            'n_classes': len(np.unique(test_labels)),
            'n_samples': len(test_labels),
            'class_proportions': np.bincount(test_labels)/len(test_labels),
        }

        # Step 6: Copy the files to the designated directories
        def copy_images(image_paths: list, destination_folder: str):
            os.makedirs(destination_folder, exist_ok=True)
            for img_path in tqdm(image_paths, desc=f"{destination_folder.split('/')[-1].capitalize()} Dataset\t"):
                class_name = img_path.split('/')[-2]  # Extract class name from path
                class_folder = f"{destination_folder}/{class_name}"
                os.makedirs(class_folder, exist_ok=True)
                shutil.copy2(img_path, class_folder)  # Copy instead of move

        # Create separate folders without modifying the original dataset
        copy_images(train_paths, f"{dst_dir}/train")
        copy_images(val_paths, f"{dst_dir}/val")
        copy_images(test_paths, f"{dst_dir}/test")

    def __augment_images(self, num_augmentations: int) -> None:
        
        dir = self.data_path
        src_dir = f"{dir}/train"
        aug_dir = f"{dir}/train_augmented"

        # Create Augmented Directory Structure
        os.makedirs(aug_dir, exist_ok=True)
        if len(os.listdir(aug_dir)) != 0:
            raise FileExistsError(f"Folders already exists within {aug_dir}.")

        for class_name in os.listdir(src_dir):
            augmented_class_path = f"{aug_dir}/{class_name}"
            os.makedirs(augmented_class_path, exist_ok=True)

        # Define augmentation pipeline
        augment = A.Compose([
            # Spatial-Level Transformations
            A.HorizontalFlip(p=0.5),  # Random horizontal flip
            A.VerticalFlip(p=0.3),  # Random vertical flip
            A.RandomRotate90(p=0.5),  # Rotate by 90-degree increments
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),  # Small shifts, scaling (±10%), and rotation (±30°)
            A.RandomResizedCrop(size=(256, 256), scale=(0.8, 1), ratio=(3.0/4.0, 4.0/3.0), p=0.3),  # Random crop & resize

            # Pixel-Level Transformations
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, ensure_safe_range=True, p=0.5),  # Adjust brightness & contrast (±20%)
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),  # Change color properties
            A.GaussianBlur(sigma_limit=(0.5, 1.5), p=0.5),  # Mild blur to simulate focus variations
        ], p=1.0)

        # Apply Augmentation
        for class_name in tqdm(os.listdir(src_dir), desc="Processing Classes"):
            class_path = f"{src_dir}/{class_name}"
            augmented_class_path = f"{aug_dir}/{class_name}"

            for img_name in os.listdir(class_path):
                img_path = f"{class_path}/{img_name}"
                image = cv2.imread(img_path)

                if image is None:
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

                # Save Original Image in Augmented Directory
                cv2.imwrite(f"{augmented_class_path}/{img_name}", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                # Generate Augmented Images
                for i in range(num_augmentations):
                    augmented = augment(image=image)["image"]
                    augmented_filename = f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg"
                    augmented_path = f"{augmented_class_path}/{augmented_filename}"
                    cv2.imwrite(augmented_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

        self.train_metadata['path'] = aug_dir

        n_samples = 0
        for class_name in os.listdir(aug_dir):
            augmented_class_path = f"{aug_dir}/{class_name}"
            n_samples += len(os.listdir(augmented_class_path))

        self.train_metadata['n_samples'] = n_samples

    def _split_augment(self, val_size, test_size, seed, num_augmentations):

        self.__train_val_test_split(val_size=val_size, test_size=test_size, seed=seed)
        print("✅ Images copied to Train, Validation, and Test folders while keeping the original dataset intact!\n")
        
        self.__augment_images(num_augmentations=num_augmentations)
        print("✅ Data Augmentation Complete!\n")

        print("\nDatasets Details:\n")
        print("Class Names:")
        for _ in self.class_names:
            print("\t", _)
        print("Class Weights:", self.class_weights)
        print()
        for _ in [self.train_metadata, self.val_metadata, self.test_metadata]:
            print(f"{_['path'].split('/')[-1].capitalize()}: {(_['split_size'])*100:.0f}%")
            print(f"No. of Classes: {_['n_classes']}")
            print(f"No. of Samples: {_['n_samples']}")
            print(f"Class Proportions: {_['class_proportions'].round(4)}")
            print()

split_augment = _Datasets()._split_augment