# Object Detection Model for Household Items

![YOLOv5 Logo](https://github.com/ultralytics/yolov5/raw/master/docs/images/yolov5-logo.png)

## Overview

This project focuses on developing a **real-time object detection model** using the **YOLO v5** (You Only Look Once, version 5) deep learning model, optimized for mobile deployment. The model identifies three specific household items—t-shirts, balls, and vacuum cleaners—and is designed to support applications such as **smart home automation** and **inventory management**.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Description](#project-description)
3. [Tasks and Workflow](#tasks-and-workflow)
4. [Data Collection & Preprocessing](#data-collection--preprocessing)
5. [Methodology](#methodology)
6. [Implementation Details](#implementation-details)
7. [Code Structure](#code-structure)
8. [Downloading YOLO v5](#downloading-yolo-v5)
9. [Results](#results)
10. [Conclusion and Future Work](#conclusion-and-future-work)

## Introduction

**Object detection** is a branch of computer vision that identifies and locates objects in digital images and videos, providing not only classification but also spatial data through bounding boxes. Unlike image classification, object detection specifies exact locations for objects, a crucial feature for applications such as **autonomous vehicles**, **security systems**, **medical imaging**, **retail inventory**, and **quality control**.

## Project Description

The aim of this project is to create an efficient, real-time **object detection system** that uses YOLO v5 to recognize t-shirts, balls, and vacuum cleaners. YOLO v5’s speed and accuracy make it ideal for this task, allowing the model to run on **mobile devices** with limited processing power. Key goals include:

- **Data Collection and Labeling**: A custom dataset was created with 100+ images for each object class, collected in various lighting conditions and settings.
- **Model Training**: We fine-tuned YOLO v5 on this dataset to optimize performance.
- **Mobile Deployment**: The model was optimized for deployment on mobile devices, providing users with instant object recognition capabilities for applications like smart home management.

## Tasks and Workflow

The project is divided into the following tasks:

1. **Data Collection and Preprocessing**
   - Collected images of t-shirts, balls, and vacuum cleaners in household settings.
   - Used data augmentation (rotation, flipping, scaling) to improve model robustness.
2. **Model Selection and Training**
   - Chose YOLO v5 for its balance of speed and accuracy.
   - Split the dataset into training, validation, and testing sets.
   - Fine-tuned the model hyperparameters.
3. **Model Deployment**
   - Deployed the trained model on mobile devices to enable real-time detection through the device's camera.

## Data Collection & Preprocessing

Images of t-shirts, balls, and vacuum cleaners were collected and labeled using **RoboFlow**. Each class had at least 100 images, ensuring variety in backgrounds and lighting conditions. We applied data augmentation techniques like rotation, flipping, and scaling to improve model generalization, making it more resilient to unseen data variations.

## Methodology

We chose **YOLO v5** due to its efficiency in real-time object detection. YOLO divides an input image into a grid, predicting bounding boxes and class probabilities simultaneously. This single-pass detection method is computationally efficient and ideal for mobile applications.

### Key Model Features
- **Single-pass Detection**: YOLO predicts object classes and bounding boxes in one pass.
- **Real-time Performance**: Optimized for high speed and low latency.
- **High Accuracy**: Effective at handling various lighting conditions and object orientations.

## Implementation Details

### Steps
1. **Data Labeling**: Annotated bounding boxes around objects using **RoboFlow**.
2. **Model Training**: Trained YOLO v5 with custom configurations, balancing model complexity with available resources.
3. **Mobile Deployment**: Optimized the trained model for mobile platforms to maintain efficiency and speed without draining battery or overloading the processor.

### Challenges
- **Hyperparameter Tuning**: Adjusting parameters like learning rate and batch size for optimal accuracy.
- **Mobile Optimization**: Ensuring model compatibility with mobile hardware for smooth performance.

## Code Structure

To set up and run the project, clone the YOLO v5 repository and follow these steps:

```bash
# Clone YOLOv5 repository
git clone https://github.com/ultralytics/yolov5
cd yolov5
# Reset to a stable commit
git reset --hard 064365d8683fd002e9ad789c1e91fa3d021b44f0
# Install dependencies
pip install -qr requirements.txt
```

### Training and Testing
After setup, the project includes scripts for training and testing:

- **`train.py`**: Train YOLO v5 on the custom dataset with specific configurations.
- **`detect.py`**: Run inference on sample images.

## Downloading YOLO v5

YOLO v5 weights and pretrained models can be downloaded directly from [Ultralytics’ YOLO v5 GitHub repository](https://github.com/ultralytics/yolov5). You can use pre-trained weights such as `yolov5s.pt`, `yolov5m.pt`, or `yolov5l.pt` as starting points or for transfer learning. To download a pre-trained model, follow these steps:

1. After cloning the YOLO v5 repository, navigate to the root folder.
2. Download the desired model using `torch.hub`:

    ```python
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    ```

3. Alternatively, download specific weights:

    ```bash
    wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
    ```

For custom training, replace these weights with your own trained weights during the training process.

## Results

### Evaluation Metrics
The model demonstrated high accuracy on t-shirts, balls, and vacuum cleaners with impressive scores on precision, recall, and **mean Average Precision (mAP)**. The results indicate:
- **High Precision**: Low false-positive rate.
- **High Recall**: Effective detection of relevant objects.
- **mAP**: Consistent performance across multiple thresholds.

### Mobile Deployment
The YOLO v5 model was successfully deployed on mobile devices, enabling real-time object detection. This opens possibilities for dynamic user interaction in smart home environments, such as inventory management, security monitoring, and augmented reality applications.

## Conclusion and Future Work

Our project achieved the goal of creating a high-accuracy, mobile-compatible object detection model. The YOLO v5-based system offers real-time performance and robust detection, making it ideal for practical applications. Future directions for improvement include:

- **Expanding Object Categories**: Adding more objects to increase versatility.
- **Optimizing Model Efficiency**: Further reducing model size and energy consumption for even more resource-limited environments.
