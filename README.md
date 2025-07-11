# CNN-CIFAR

## Image Classification using Convolutional Neural Networks on CIFAR-10

This repository contains code and resources for training and evaluating Convolutional Neural Networks (CNNs) on the CIFAR-10 dataset.

### Features

- Implementation of a plain CNN
- Implementation of ResNet architecture
- Training and evaluation scripts
- Support for data augmentation and preprocessing
- Visualization of training metrics

### Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) is a dataset of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

### Getting Started

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/CNN-CIFAR.git
    cd CNN-CIFAR
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Train the model:**
    ```bash
    python train.py
    ```

4. **Test the model:**
    ```bash
    python test.py
    ```

### Project Structure

```
CNN-CIFAR/
├── model/           # CNN model definitions
├── train.py         # Training script
├── test.py          # Evaluation script
├── dataloader.py    # Load data and download CIFAR-10 dataset if not present
└── README.md
```

### Results

| Model        | Accuracy |
|--------------|----------|
| Simple CNN   | NA       |
| Advanced CNN | NA       |

### References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Dive Into Deep Learning](https://d2l.ai)

