import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_dataset_from_folders(dataset_dir: str, use_test_split: bool = True) -> Tuple[List[np.ndarray], List[str], Dict[str, int]]:
	"""
	Load images from folder structure (supports two formats):
	
	Format 1 (train/test structure):
	dataset/
	  train/
	    happy/
	    sad/
	    angry/
	    neutral/
	  test/
	    happy/
	    sad/
	    angry/
	    neutral/
	
	Format 2 (flat structure):
	dataset/
	  happy/
	  sad/
	  angry/
	  neutral/
	
	If use_test_split=True and train/test structure exists, uses train for training.
	Otherwise, loads all images from the structure found.
	
	Returns: (images, labels, idx_to_label)
	"""
	dataset_path = Path(dataset_dir)
	if not dataset_path.exists():
		raise ValueError(f"Dataset directory not found: {dataset_dir}")
	
	images = []
	labels = []
	label_to_idx = {}
	
	# Check if train/test structure exists
	train_path = dataset_path / "train"
	test_path = dataset_path / "test"
	
	has_train_test = train_path.exists() and test_path.exists()
	
	if has_train_test:
		print("Found train/test folder structure")
		# Always use train folder to discover emotion classes
		class_dirs = [d for d in train_path.iterdir() if d.is_dir()]
		class_dirs.sort()
		
		# Build label mapping from train folder
		for idx, class_dir in enumerate(class_dirs):
			label_name = class_dir.name.lower()
			label_to_idx[label_name] = idx
		
		idx_to_label = {v: k for k, v in label_to_idx.items()}
		print(f"Found {len(class_dirs)} emotion classes: {list(label_to_idx.keys())}")
		
		# Load from train folder
		print("\nLoading images from 'train' folder...")
		for class_dir in class_dirs:
			label_name = class_dir.name.lower()
			label_idx = label_to_idx[label_name]
			
			image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + \
			              list(class_dir.glob("*.png")) + list(class_dir.glob("*.bmp"))
			
			print(f"  Loading {len(image_files)} images from train/{label_name}...")
			
			for img_path in image_files:
				try:
					img = Image.open(img_path)
					if img.mode != "RGB":
						img = img.convert("RGB")
					img = img.resize((224, 224))
					img_array = np.array(img, dtype=np.float32) / 255.0
					images.append(img_array)
					labels.append(label_idx)
				except Exception as e:
					print(f"Warning: Could not load {img_path}: {e}")
					continue
		
		# Optionally load from test folder if not using test split
		if not use_test_split:
			print("\nLoading images from 'test' folder...")
			for class_dir in class_dirs:
				label_name = class_dir.name.lower()
				if label_name not in label_to_idx:
					continue
				label_idx = label_to_idx[label_name]
				
				# Load from corresponding test subfolder
				test_class_dir = test_path / class_dir.name
				if not test_class_dir.exists():
					continue
				
				image_files = list(test_class_dir.glob("*.jpg")) + list(test_class_dir.glob("*.jpeg")) + \
				              list(test_class_dir.glob("*.png")) + list(test_class_dir.glob("*.bmp"))
				
				print(f"  Loading {len(image_files)} images from test/{label_name}...")
				
				for img_path in image_files:
					try:
						img = Image.open(img_path)
						if img.mode != "RGB":
							img = img.convert("RGB")
						img = img.resize((224, 224))
						img_array = np.array(img, dtype=np.float32) / 255.0
						images.append(img_array)
						labels.append(label_idx)
					except Exception as e:
						print(f"Warning: Could not load {img_path}: {e}")
						continue
	else:
		# Flat structure: load directly from dataset root
		print("Found flat folder structure (no train/test subfolders)")
		class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
		class_dirs.sort()
		
		# Build label mapping
		for idx, class_dir in enumerate(class_dirs):
			label_name = class_dir.name.lower()
			label_to_idx[label_name] = idx
		
		idx_to_label = {v: k for k, v in label_to_idx.items()}
		print(f"Found {len(class_dirs)} emotion classes: {list(label_to_idx.keys())}")
		
		# Load images
		for class_dir in class_dirs:
			label_name = class_dir.name.lower()
			label_idx = label_to_idx[label_name]
			
			image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + \
			              list(class_dir.glob("*.png")) + list(class_dir.glob("*.bmp"))
			
			print(f"Loading {len(image_files)} images from '{label_name}'...")
			
			for img_path in image_files:
				try:
					img = Image.open(img_path)
					if img.mode != "RGB":
						img = img.convert("RGB")
					img = img.resize((224, 224))
					img_array = np.array(img, dtype=np.float32) / 255.0
					images.append(img_array)
					labels.append(label_idx)
				except Exception as e:
					print(f"Warning: Could not load {img_path}: {e}")
					continue
	
	images = np.array(images)
	labels = np.array(labels)
	
	print(f"\nTotal images loaded: {len(images)}")
	if len(labels) > 0:
		print(f"Label distribution: {np.bincount(labels)}")
	
	return images, labels, idx_to_label


def build_model(num_classes: int) -> keras.Model:
	"""
	Build MobileNetV2-based model with custom classification head.
	"""
	# Load pre-trained MobileNetV2 (ImageNet weights)
	base_model = MobileNetV2(
		weights="imagenet",
		include_top=False,
		input_shape=(224, 224, 3)
	)
	
	# Freeze base model layers (optional: can unfreeze last few layers for fine-tuning)
	base_model.trainable = False
	
	# Add custom classification head
	model = models.Sequential([
		base_model,
		layers.GlobalAveragePooling2D(),
		layers.Dropout(0.2),
		layers.Dense(128, activation="relu"),
		layers.Dropout(0.2),
		layers.Dense(num_classes, activation="softmax")
	])
	
	model.compile(
		optimizer=keras.optimizers.Adam(learning_rate=0.001),
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"]
	)
	
	return model


def train_model_from_directories(
	train_dir: str,
	test_dir: Optional[str],
	idx_to_label: Dict[int, str],
	output_model_path: str,
	output_labels_path: str,
	epochs: int = 10,
	batch_size: int = 32,
	validation_split: float = 0.1
):
	"""
	Train the model using directory-based loading (memory efficient).
	Uses ImageDataGenerator.flow_from_directory to load images in batches.
	"""
	num_classes = len(idx_to_label)
	model = build_model(num_classes)
	
	print("\nModel architecture:")
	model.summary()
	
	# Data augmentation for training
	train_datagen = ImageDataGenerator(
		rescale=1.0 / 255.0,
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		horizontal_flip=True,
		zoom_range=0.2,
		validation_split=validation_split
	)
	
	# Validation generator (no augmentation, just rescaling)
	val_datagen = ImageDataGenerator(
		rescale=1.0 / 255.0,
		validation_split=validation_split
	)
	
	# Training generator
	print("\nSetting up training data generator...")
	train_generator = train_datagen.flow_from_directory(
		train_dir,
		target_size=(224, 224),
		batch_size=batch_size,
		class_mode='sparse',
		subset='training',
		shuffle=True,
		seed=42
	)
	
	# Validation generator
	val_generator = val_datagen.flow_from_directory(
		train_dir,
		target_size=(224, 224),
		batch_size=batch_size,
		class_mode='sparse',
		subset='validation',
		shuffle=False,
		seed=42
	)
	
	# Calculate steps per epoch
	steps_per_epoch = train_generator.samples // batch_size
	validation_steps = val_generator.samples // batch_size
	
	print(f"Training samples: {train_generator.samples}")
	print(f"Validation samples: {val_generator.samples}")
	print(f"Steps per epoch: {steps_per_epoch}")
	
	# Train the model
	print("\nTraining model...")
	history = model.fit(
		train_generator,
		steps_per_epoch=steps_per_epoch,
		epochs=epochs,
		validation_data=val_generator,
		validation_steps=validation_steps,
		verbose=1
	)
	
	# Evaluate on test set if available
	if test_dir and Path(test_dir).exists():
		print("\nEvaluating on test set...")
		test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
		test_generator = test_datagen.flow_from_directory(
			test_dir,
			target_size=(224, 224),
			batch_size=batch_size,
			class_mode='sparse',
			shuffle=False
		)
		
		test_loss, test_acc = model.evaluate(test_generator, verbose=0)
		print(f"Test Accuracy: {test_acc:.4f}")
		
		# Get predictions for classification report
		test_generator.reset()
		y_pred = model.predict(test_generator, verbose=0)
		y_pred_classes = np.argmax(y_pred, axis=1)
		
		# Get true labels
		y_true = test_generator.classes
		
		print("\nClassification Report:")
		print(classification_report(
			y_true, y_pred_classes,
			target_names=[idx_to_label[i] for i in range(num_classes)]
		))
	
	# Save model
	print(f"\nSaving model to {output_model_path}")
	model.save(output_model_path)
	
	# Save label mapping
	print(f"Saving labels to {output_labels_path}")
	with open(output_labels_path, "w") as f:
		json.dump(idx_to_label, f, indent=2)
	
	print("Training complete!")


def train_model(
	images: np.ndarray,
	labels: np.ndarray,
	idx_to_label: Dict[int, str],
	output_model_path: str,
	output_labels_path: str,
	test_size: float = 0.2,
	epochs: int = 10,
	batch_size: int = 32,
	validation_split: float = 0.1
):
	"""
	Train the model and save it along with label mapping.
	Legacy function for in-memory training (use train_model_from_directories for large datasets).
	"""
	print("Splitting dataset...")
	X_train, X_test, y_train, y_test = train_test_split(
		images, labels, test_size=test_size, random_state=42, stratify=labels
	)
	
	print(f"Train: {len(X_train)}, Test: {len(X_test)}")
	
	# Data augmentation for training
	train_datagen = ImageDataGenerator(
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		horizontal_flip=True,
		zoom_range=0.2,
		validation_split=validation_split
	)
	
	num_classes = len(idx_to_label)
	model = build_model(num_classes)
	
	print("\nModel architecture:")
	model.summary()
	
	print("\nTraining model...")
	history = model.fit(
		X_train, y_train,
		batch_size=batch_size,
		epochs=epochs,
		validation_split=validation_split,
		verbose=1
	)
	
	print("\nEvaluating on test set...")
	test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
	print(f"Test Accuracy: {test_acc:.4f}")
	
	# Predictions for classification report
	y_pred = model.predict(X_test, verbose=0)
	y_pred_classes = np.argmax(y_pred, axis=1)
	
	print("\nClassification Report:")
	print(classification_report(
		y_test, y_pred_classes,
		target_names=[idx_to_label[i] for i in range(num_classes)]
	))
	
	# Save model
	print(f"\nSaving model to {output_model_path}")
	model.save(output_model_path)
	
	# Save label mapping
	print(f"Saving labels to {output_labels_path}")
	with open(output_labels_path, "w") as f:
		json.dump(idx_to_label, f, indent=2)
	
	print("Training complete!")


def main():
	parser = argparse.ArgumentParser(description="Train image emotion classifier (MobileNetV2)")
	parser.add_argument(
		"--dataset",
		default=os.path.join(os.path.dirname(__file__), "dataset"),
		help="Path to dataset folder with emotion subfolders"
	)
	parser.add_argument(
		"--model",
		default=os.path.join(os.path.dirname(__file__), "image_model.h5"),
		help="Output path for trained model"
	)
	parser.add_argument(
		"--labels",
		default=os.path.join(os.path.dirname(__file__), "labels.json"),
		help="Output path for label mapping JSON"
	)
	parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
	parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
	parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
	parser.add_argument("--val-split", type=float, default=0.1, help="Validation split")
	parser.add_argument(
		"--use-all-data",
		action="store_true",
		help="If dataset has train/test folders, load from both (default: only train)"
	)
	
	args = parser.parse_args()
	
	dataset_path = Path(args.dataset)
	train_path = dataset_path / "train"
	test_path = dataset_path / "test"
	
	# Check if train/test structure exists
	has_train_test = train_path.exists() and test_path.exists()
	
	if has_train_test:
		# Use directory-based training (memory efficient)
		print("Using directory-based training (memory efficient)...")
		
		# Discover emotion classes from train folder
		class_dirs = [d for d in train_path.iterdir() if d.is_dir()]
		class_dirs.sort()
		
		label_to_idx = {}
		for idx, class_dir in enumerate(class_dirs):
			label_name = class_dir.name.lower()
			label_to_idx[label_name] = idx
		
		idx_to_label = {v: k for k, v in label_to_idx.items()}
		print(f"Found {len(class_dirs)} emotion classes: {list(label_to_idx.keys())}")
		
		# Determine which directories to use
		train_dir = str(train_path)
		test_dir = str(test_path) if (test_path.exists() and not args.use_all_data) else None
		
		# If use_all_data, we still use train for training but can evaluate on test
		if args.use_all_data:
			test_dir = str(test_path) if test_path.exists() else None
		
		train_model_from_directories(
			train_dir=train_dir,
			test_dir=test_dir,
			idx_to_label=idx_to_label,
			output_model_path=args.model,
			output_labels_path=args.labels,
			epochs=args.epochs,
			batch_size=args.batch_size,
			validation_split=args.val_split
		)
	else:
		# Fallback to in-memory training for flat structure
		print("Loading dataset from folders (in-memory)...")
		images, labels, idx_to_label = load_dataset_from_folders(
			args.dataset,
			use_test_split=not args.use_all_data
		)
		
		if len(images) == 0:
			raise ValueError("No images found in dataset directory!")
		
		train_model(
			images, labels, idx_to_label,
			args.model, args.labels,
			test_size=args.test_size,
			epochs=args.epochs,
			batch_size=args.batch_size,
			validation_split=args.val_split
		)


if __name__ == "__main__":
	main()

