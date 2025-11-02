## Tech Arena 2025 - Anthropometric Data Extraction

Dataset and models for extracting anthropometric data of the human head and ears from a series of RGBD images. This submission implements deep learning models to extract specific anthropometric measurements from HEIC format images taken from various angles around human subjects.

The goal of this challenge is to extract anthropometric data of the human head and ears from a series of RGBD images of a human subject. Anthropometric data refers to measurements of the human anatomy, in this case the human head and ears. The images of the human head from which the anthropometric measurements should be extracted are taken from various angles around the subject and include both RGB data as well as depth information.

This dataset release includes code to train models on SONICOM dataset, to use trained model checkpoints for inference and to evaluate extracted results on test data. Pretrained models used in this submission are included so results can be easily reproduced without training the model checkpoints yourself.

Instructions and code can be found in this github repository. Training code can be found in the `train_2.py` script. Source code is provided in `src_1.zip` and pretrained models are available in `models.zip`.

The submitted models are expected to accept a fixed number of subject images taken from predefined camera positions and to output specific anthropometric measurements of the human head and pinnas.

The model works with 3 different sets of input images:
1. images from all directions (72 images)
2. front images only (36 images)
3. only front, left and right images (3 images)

Data sourced from the [SONICOM](https://www.sonicom.eu/) dataset. Original images sourced with permission from the SONICOM project for research purposes.

| Input Type | Images | Description |
| ---------- | ------ | ----------- |
| All directions | 72 | Full circle capture at 5 degree resolution |
| Front only | 36 | Front-facing images only |
| Minimal set | 3 | Front, left, and right images |

This work is part of the Tech Arena 2025 challenge and implements state-of-the-art deep learning approaches for anthropometric measurement extraction from RGBD imagery.

# Installation under Ubuntu 20.04LTS

```

# install conda see https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html
# Download and install Anaconda or Miniconda for Windows

# create anthro_env in conda
conda create -n anthro_env python=3.10 -y
conda activate anthro_env

# check you have python 3.10 via conda
python -V

# install pytorch and dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2" "opencv-python<4.10" matplotlib
conda install -c conda-forge exiftool
pip install pyexiftool
conda install -c conda-forge -y libheif pillow-heif
pip install tensorboardX
pip install pandas
pip install ultralytics
conda install jupyter -y

# extract source code
unzip src_1.zip

# extract pretrained models
unzip models.zip

# prepare the training data files
# Note: SONICOM dataset access requires signed data sharing permission form
# Contact organizers for access to full training dataset
```

# Train models

This is only needed if you are not using the available pretrained model checkpoints.

```
conda activate anthro_env

#
# Train >> Anthropometric Extraction Models
#

# Stage 1 Training - SVCNN RGBD Model
python train_2.py --stage 1 --model svcnn_rgbd --epochs 100 --batch_size 16 --lr 0.001

# Stage 2 Training - AnthropometricViewGCN Model
python train_2.py --stage 2 --model viewgcn --epochs 50 --batch_size 8 --lr 0.0005

# Ensemble Training - Multiple view aggregation
python train_2.py --model ensemble --epochs 75 --batch_size 12 --lr 0.0008

# Fine-tuning on specific image configurations
python train_2.py --config all_directions --epochs 25 --lr 0.0001
python train_2.py --config front_only --epochs 25 --lr 0.0001
python train_2.py --config minimal_set --epochs 25 --lr 0.0001

# resume training if needed from a specific epoch
python train_2.py --resume_from checkpoints/epoch_50.pth --epochs 100
```

In order to change more advanced training settings, one has to edit the training configuration in `train_2.py`.
To change the total epoch number, please edit the epochs parameter in the training arguments.
To change the learning rate and optimizer settings, edit the optimizer configuration in the training script.
Full documentation of the training options can be found in the source code documentation.

To apply data augmentation, the training script includes RGB-specific transforms and Gaussian noise addition for improved model robustness.

To reduce memory footprint during training, one can adjust batch sizes and use gradient checkpointing options available in the model configurations.

# Inference and evaluation using models

Commands provided for using both the available pretrained models and ones trained using previous section.

```
conda activate anthro_env

#
# Inference >> Pretrained Models >> Anthropometric Extraction
#

# Run inference using pretrained ensemble model
python -c "
from extractor import AnthropometricExtractor
import numpy as np

# Load pretrained extractor
extractor = AnthropometricExtractor()
extractor.load_models('models/')

# Example inference on test images
test_images = 'path/to/test/images'
results = extractor.extract_anthropometrics(test_images)
print('Extracted measurements:', results)
"

# Run evaluation on test set
python evaluate.py --model_path models/ --test_data path/to/test --output results/

# Evaluate on different image configurations
python evaluate.py --config all_directions --test_data path/to/test
python evaluate.py --config front_only --test_data path/to/test
python evaluate.py --config minimal_set --test_data path/to/test

# Run the getting_started.ipynb for complete evaluation pipeline
jupyter notebook getting_started.ipynb
```

# Evaluation Metrics

The submitted systems are evaluated using Euclidean distance between standardized feature vectors. For an N-dimensional feature vector, standardization is achieved independently for each feature dimension by subtracting its mean and dividing by its standard deviation.

The Euclidean distance between the standardized output and the standardized ground truth is computed as the primary evaluation metric. The overall score is computed as the mean across all subjects and input image sets. Since the applied metric is a distance, a lower score corresponds to a better result.

# Submission

The final submission includes:
1. `src_1.zip` - the source code to extract anthropometric parameters from RGBD images
2. `models.zip` - pretrained model checkpoints for inference
3. `train_2.py` - training code and reference implementation
4. this README documentation

Each submitted model implements the `Extractor` class in the `extractor.py` module. This class is used for automatic evaluation on the hidden test data set.

# Background

*Binaural audio rendering* is the process of simulating sound sources in 3D space around a listener. It is not only used for virtual reality and augmented reality applications, but has made its way into mobile devices to provide immersive user experiences when listening to audio content.

In order to provide the illusion of sounds coming from various directions, sound source signals are convolved with so-called *head-related transfer functions (HRTFs)*. Those HRTFs encode the relevant binaural cues that let the listener perceive the sound from a certain direction.

However, HRTFs are influenced by the human anatomy. The pinna causes direction-dependent sound reflections and the head causes frequency-dependent sound attenuation due to shadowing effects. Therefore, HRTFs of individuals differ due to anatomic differences.

The goal of this challenge is therefore to obtain these anthropometric measurements from a set of RGBD images for personalized HRTF simulation.

# Anthropometrics

The provided anthropometrics are based on landmark points on the subject's head and pinna. Each dimension in the anthropometric feature vector represents the Euclidean distance between two landmark points.

The anthropometrics are defined as:
- x₁ = |IJ| (distance between ear canal entrances)
- p₁ = |BC| (tragus to intertragal notch distance)
- p₂ = |AD| (tragus to antihelix distance)
- p₃ = |CF| (antihelix to helix top distance)
- p₄ = |GH| (helix top to earlobe distance)
- p₅ = |AE| (tragus to outer helix distance)

The overall anthropometrics feature vector is formed as:
**a** = [x₁, p₁,left, ..., p₅,left, p₁,right, ..., p₅,right]ᵀ

# Model Architecture

## Key Features:
- SVCNN (Single View CNN) for RGBD image processing
- AnthropometricViewGCN for multi-view feature aggregation
- SVCNNEnsemble for robust measurement extraction
- Support for variable input configurations (3, 36, or 72 images)

## Technical Implementation:
- PyTorch-based deep learning models
- HEIC image format support via libheif and pillow-heif
- Metadata extraction using PyExifTool
- Standardized evaluation metrics implementation
