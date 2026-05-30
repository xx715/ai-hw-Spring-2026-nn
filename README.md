# MNIST Classification and Adversarial Robustness

This project explores both image classification performance and adversarial robustness on the MNIST handwritten digit dataset using three neural network architectures:

* Multi-Layer Perceptron (MLP)
* Convolutional Neural Network (CNN)
* Vision Transformer (ViT)

The project consists of two parts:

1. Comparing classification performance across architectures.
2. Evaluating adversarial robustness using FGSM, I-FGSM, and MI-FGSM attacks.

---

## Repository Structure

```text
.
├── mnist_homework copy.ipynb
│   ├── Data loading and preprocessing
│   ├── MLP implementation
│   ├── CNN implementation
│   ├── Vision Transformer implementation
│   └── Data augmentation experiment
│
└── mnist_adversarial_attacks.ipynb
    ├── Model training
    ├── FGSM attack
    ├── I-FGSM attack
    ├── MI-FGSM attack
    ├── Attack Success Rate evaluation
    └── Robustness visualizations
```

---

# Part 1: MNIST Image Classification

## Dataset

The experiments use the MNIST handwritten digit dataset:

* 60,000 training images
* 10,000 test images
* Grayscale images
* Resolution: 28 × 28 pixels
* 10 digit classes (0–9)

Pixel values are normalized from:

```text
[0, 255] → [0, 1]
```

---
## Training Configuration

All models were trained using the same settings:

| Parameter     | Value              |
| ------------- | ------------------ |
| Optimizer     | Adam               |
| Learning Rate | 0.001              |
| Batch Size    | 64                 |
| Epochs        | 5                  |
| Loss Function | Cross Entropy Loss |

Using identical training settings helps provide a fair comparison between architectures.



---

## Model Architectures

### Multi-Layer Perceptron (MLP)

The MLP serves as a simple baseline model.

Architecture:

```text
Input (784)
    ↓
Linear(784 → 128)
    ↓
ReLU
    ↓
Linear(128 → 10)
```

Characteristics:

* Fully connected network
* Uses flattened image vectors
* Does not explicitly preserve spatial information

---

### Convolutional Neural Network (CNN)

The CNN is designed specifically for image classification.

Architecture:

```text
28×28 Image
    ↓
Conv2D(16 filters)
    ↓
Max Pooling
    ↓
Conv2D(32 filters)
    ↓
Max Pooling
    ↓
Fully Connected Layer
    ↓
Output (10 classes)
```

Characteristics:

* Learns local spatial features
* Parameter sharing through convolutions

---

### Transformer

The Transformer treats images as sequences of patches.

Configuration:

```text
Patch Size: 7 × 7
Number of Patches: 16
Embedding Dimension: 64
Attention Heads: 4
Transformer Layers: 2
```

Pipeline:

```text
Image
    ↓
Patch Embedding
    ↓
Positional Encoding
    ↓
Transformer Encoder
    ↓
Mean Pooling
    ↓
Linear Classifier
```
---

## Data Augmentation Experiment

An additional CNN experiment was performed using image augmentation.

Augmentations:

```python
RandomRotation(10)
RandomAffine(
    degrees=0,
    translate=(0.1, 0.1)
)
```

Purpose:

* Simulate handwriting variations
* Improve generalization
* Increase robustness to small transformations
---



## Classification Results

| Model              | Test Accuracy |
| ------------------ | ------------- |
| CNN                | 99.06%        |
| MLP                | 97.41%        |
| Vision Transformer | 96.66%        |
| CNN + Augmentation | 98.87%        |
<img width="577" height="468" alt="Screenshot 2026-05-30 at 10 49 01 AM" src="https://github.com/user-attachments/assets/56593f89-9ee5-4664-9e5e-32e126492240" />

### Observations

* CNN achieved the highest classification accuracy.
* MLP achieved strong performance despite using a much simpler architecture.
* Transformer remained competitive, but slightly underperformed the CNN in this experiment as transformers generally benefit from larger datasets and longer training.
* Data augmentation resulted in a small decrease in test accuracy. Since the MNIST test set is already clean and well-aligned, augmentation introduces variability that is not reflected in the test data.

---

# Part 2: Adversarial Attacks

## Objective

The second part of the project evaluates how vulnerable each architecture is to adversarial examples.

An adversarial example is a modified input image that appears visually similar to a human but causes the model to make an incorrect prediction.

---

## Attack Methods

### FGSM (Fast Gradient Sign Method)

Single-step attack:

```math
x_{adv} = x + \epsilon \cdot sign(\nabla_x L)
```

Characteristics:

* Fast
* Computationally inexpensive
* Baseline adversarial attack

---

### I-FGSM (Iterative FGSM)

Performs multiple small FGSM updates.

Characteristics:

* Stronger than FGSM
* Iteratively refines perturbations
* Constrained by the ε-budget

---

### MI-FGSM (Momentum Iterative FGSM)

Extends I-FGSM by incorporating momentum.

Characteristics:

* Stabilizes optimization direction
* Produces more transferable adversarial examples
* Often achieves higher attack success rates

---

## Evaluation Metric

### Attack Success Rate (ASR)

ASR measures how often an attack changes a previously correct prediction into an incorrect one.

```math
ASR =
\frac{
\text{Correct on clean data and incorrect after attack}
}{
\text{Correct on clean data}
}
```

Higher ASR indicates a more effective attack.
<img width="1043" height="499" alt="Screenshot 2026-05-30 at 10 47 18 AM" src="https://github.com/user-attachments/assets/27b36631-0583-413d-9f15-84236928114d" />
<img width="1590" height="515" alt="image" src="https://github.com/user-attachments/assets/7929a711-3e47-4908-8445-ecfe7f50b8fc" />

---

## Experimental Setup

Perturbation budgets:

```python
EPSILONS = [0.05, 0.10, 0.20, 0.30]
```

Models evaluated:

* MLP
* CNN
* Vision Transformer

Attacks evaluated:

* FGSM
* I-FGSM
* MI-FGSM

---

## Key Findings

### MLP

* Most vulnerable architecture
* High attack success rates even at small perturbation levels

### CNN

* Most robust architecture at low ε values
* Strong resistance to FGSM at ε = 0.05
* Vulnerability increases substantially under iterative attacks

### Vision Transformer

* Robustness generally falls between MLP and CNN
* Better than MLP
* Less robust than CNN at lower perturbation levels

### Overall

* Attack success rate increases as ε increases.
* Iterative attacks consistently outperform FGSM.
* At sufficiently large perturbation budgets, all architectures become highly vulnerable.

---

## Visualizations

The adversarial attack notebook generates:

### ASR vs Epsilon Curves

Shows how attack effectiveness changes as perturbation strength increases.

### Heatmaps

Provides a model-by-attack comparison across all epsilon values.

### Adversarial Examples

Displays original and perturbed images for qualitative analysis.

---

## Dependencies

```bash
pip install torch torchvision
pip install pandas numpy
pip install matplotlib seaborn
pip install pillow
pip install pyarrow
```

---

## Running the Notebooks

### Classification Experiments

```bash
jupyter notebook "mnist_homework copy.ipynb"
```

### Adversarial Robustness Experiments

```bash
jupyter notebook "mnist_adversarial_attacks.ipynb"
```

---

## Summary

This project compares MLP, CNN, and Vision Transformer architectures on MNIST and evaluates their robustness against adversarial attacks.

Results show that:

* CNN achieves the highest classification accuracy.
* CNN demonstrates the strongest robustness at small perturbation levels.
* Iterative attacks are significantly more effective than single-step attacks.
* Increasing perturbation strength eventually makes all architectures vulnerable.

Together, these experiments highlight the tradeoff between model performance and robustness in deep learning systems.
