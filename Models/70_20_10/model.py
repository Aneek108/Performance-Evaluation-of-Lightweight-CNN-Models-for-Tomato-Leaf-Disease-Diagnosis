from os import listdir
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import data
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV3Small, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_v3_preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.layers import Dense, Rescaling, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalFocalCrossentropy

class _model:

    def __init__(self):
        self.split_paradigm = None

    def _load_tf_datasets(self, split_paradigm: Literal["70_20_10", "80_10_10", "70_15_15"]) -> tuple[data.Dataset, data.Dataset, data.Dataset]:
        
        datasets_dir = f"E:/Research & Projects/Tomato Disease Detection/Saved Dataset/Splited/{split_paradigm}"
        train_path = f"{datasets_dir}/train_augmented"
        val_path = f"{datasets_dir}/val"
        test_path = f"{datasets_dir}/test"
        self.split_paradigm = split_paradigm

        params = dict(
            labels='inferred',
            label_mode="categorical",
            class_names=sorted(listdir(train_path)),
            color_mode='rgb',
            batch_size=32,
            image_size=(224, 224),
            shuffle=True,
            seed=42,
            verbose=False
        )

        train_ds = image_dataset_from_directory(directory=train_path, **params)
        val_ds = image_dataset_from_directory(directory=val_path, **params)
        test_ds = image_dataset_from_directory(directory=test_path, **params)

        for _ in [train_ds, val_ds, test_ds]:
            print(type(_))
            print(f"n_samples: {_.file_paths.__len__()}")
            print(f"n_batches: {_.cardinality().numpy()}")
            print(f"n_classes: {_.class_names.__len__()}")
            print(f"Classes: {_.class_names}")
            x, y = _.take(1).as_numpy_iterator().next()
            print(f"Image Batch Shape: {x.shape}")
            print(f"Label Batch Shape: {y.shape}")
            print()

        cache_dir = f"E:/Research & Projects/Tomato Disease Detection/Models/{split_paradigm}/cache"
        train_ds = train_ds.cache(f"{cache_dir}/train_ds").shuffle(200, reshuffle_each_iteration=True).prefetch(data.AUTOTUNE)
        val_ds = val_ds.cache(f"{cache_dir}/val_ds").prefetch(data.AUTOTUNE)
        test_ds = test_ds.cache(f"{cache_dir}/test_ds").prefetch(data.AUTOTUNE)

        return train_ds, val_ds, test_ds

    def _build_model(self, model_name: Literal["Custom_CNN", "MobileNetV3Small", "EfficientNetB0"]) -> Model:

        if model_name == "Custom_CNN":
            model = Sequential([
                # Hidden Layers not disclosed. 
                
                Dense(10, activation='softmax')  # Softmax for multi-class classification
            ], name=model_name)

            model.summary()
            return model
        
        else:
            base_model_class_preprocess = {
                "MobileNetV3Small": (MobileNetV3Small, mobilenet_v3_preprocess_input),
                "EfficientNetB0": (EfficientNetB0, efficientnet_preprocess_input),
            }
            
            base_model = base_model_class_preprocess[model_name][0](include_top=False, input_shape=(224, 224, 3), weights='imagenet')
            base_model.trainable = False

            input = Input(shape=(224, 224, 3), dtype="float32")
            x = base_model_class_preprocess[model_name][1](input)
            x = base_model(x)
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dense(64, activation='relu')(x)
            x = Dense(10, activation='softmax')(x) # Softmax for multi-class classification

            model = Model(inputs=input, outputs=x, name=model_name)
            model.summary()
            return model
            
    def _compile_train_model(self, model, train_ds: data.Dataset, val_ds: data.Dataset) -> Model:

        # Derived from sklearn.utils.class_weight.compute_class_weight using train_labels
        class_weights = np.array([
            0.85378467, 1.816, 1.14142049, 0.95128339, 1.90756303, 4.86863271, 1.02540937, 1.08353222, 1.29344729, 0.33899571
        ])
        alpha_values = class_weights / np.max(class_weights)  # Scale relative to max class weight

        model.compile(optimizer=Adam(), loss=CategoricalFocalCrossentropy(alpha=alpha_values), metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

        history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[early_stopping, reduce_lr])
        model_save_dir = f"E:/Research & Projects/Tomato Disease Detection/Models/{self.split_paradigm}/Saved Models"
        model.save(f"{model_save_dir}/{model.name}.keras")

        # Extract values from history
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        epochs = range(1, len(train_loss) + 1)

        plt.figure(figsize=(12, 5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, 'r-', label='Training Loss')
        plt.plot(epochs, val_loss, 'b-', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.legend()

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc, 'r-', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'b-', label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training & Validation Accuracy')
        plt.legend()

        plt.show()

        return model

    def _evaluate_model(self, model, test_ds: data.Dataset) -> None:

        test_loss, test_acc = model.evaluate(test_ds)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        # Get true labels and predictions
        y_true = np.concatenate([y for x, y in test_ds], axis=0)
        y_pred_probs = model.predict(test_ds)  # Get predicted probabilities
        y_pred = np.argmax(y_pred_probs, axis=1)  # Convert to class indices
        y_true = np.argmax(y_true, axis=1)  # Convert one-hot to class indices

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        class_names = [
            "Bacterial_spot",
            "Early_blight",
            "Healthy",
            "Late_blight",
            "Leaf_Mold",
            "Mosaic_virus",
            "Septoria_leaf_spot",
            "Spider_mites",
            "Target_Spot",
            "Yellow_Leaf_Curl_Virus"
        ]

        # Plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix')
        plt.show()

        print(classification_report(y_true, y_pred))

m = _model()
load_tf_datasets = m._load_tf_datasets
build_model = m._build_model
compile_train_model = m._compile_train_model
evaluate_model = m._evaluate_model
