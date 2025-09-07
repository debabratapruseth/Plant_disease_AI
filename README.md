# A beginner-friendly deep learning project to detect plant disease

## Project Overview

Plant disease detection using machine learning can help farmers diagnose problems earlier, protect yields, and improve food safety for communities. This project uses the PlantVillage dataset and fine-tunes ImageNet-pretrained EfficientNet backbones (B0 and B4) to build a multi-class leaf disease classifier.

We start with EfficientNet-B0 as a baseline and then improve performance by switching to EfficientNet-B4 and using techniques such as focal loss, two-stage fine-tuning, and mixed precision. The code includes evaluation and explainability (Grad-CAM) so you can inspect what the model is “looking at.”

Why this is useful: early detection speeds up treatment decisions, reduces crop loss, and supports sustainable farming.


## Dataset

Dataset → PlantVillage (via tensorflow_datasets)

	•	Total images: ~54,000 (public PlantVillage collection)
	•	Classes: 38 categories (labels like Tomato___Late_blight, Apple___Black_rot, Potato___Late_blight, etc.)



## Features

	•	Transfer learning with EfficientNet-B0 and EfficientNet-B4 backbones (ImageNet weights)
	•	Two-stage training: (1) train a classifier head with the base frozen, (2) unfreeze top layers and fine-tune
	•	Class-imbalance handling with focal loss
	•	Efficient tf.data.Dataset pipeline and lightweight augmentations for speed and robustness
	•	Mixed-precision training support for faster runs on modern GPUs
	•	Evaluation outputs: per-class classification report, confusion matrix, and top-k accuracy


## Evaluation (high-level)

	•	The B4 experiment (larger backbone + focal loss + two-stage fine-tuning) produced clear improvements over the baseline B0 experiment.

 ### Improvement ideas (things you can try further to increase model efficiency)
 
	•	Train specialist models per plant species and ensemble their predictions.
	•	Try stronger augmentations (random crop, color jitter, cutout) and longer fine-tuning schedules.
	•	Clean and curate the dataset to remove images with spurious background cues.
	•	Experiment with other backbones (EfficientNetV2, ViT) or alternative imbalance strategies (class-balanced loss, oversampling).
	•	Use Grad-CAM across true positives, false positives and false negatives to diagnose failure modes.

## Code Structure

	1.	Environment Setup & Dependencies
	2.	Dataset Downloading & Preparation
	3.	Exploratory Data Analysis (EDA)
	4.	Configuration & Constants
	5.	Train-Validation Split
	6.	TensorFlow Data Pipeline (tf.data)
	7.	Model Building (EfficientNet-B0 / EfficientNet-B4)
	8.	Callbacks & Class Weights / Focal Loss
	9.	Model Training: Stage 1 & Stage 2
	10.	Evaluation & Metrics
	11.	Training Curves Visualization
	12.	Making Predictions on New Images
	13.	Explainability: Grad-CAM


## How to use
	1.	Open the notebook or Colab and run the cells in order (setup → data → model → train → evaluate).
	2.	If using Colab, enable GPU and install dependencies from requirements.txt.


## License

This project is licensed under the MIT License — you’re free to use, modify, and distribute it.
