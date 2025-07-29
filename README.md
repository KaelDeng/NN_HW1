# CIFAR-10 Classifier - Three-Layer Neural Network

This project implements a three-layer neural network classifier for image classification on the CIFAR-10 dataset. The entire network, including forward and backward propagation, is implemented from scratch using NumPy, without any automatic differentiation frameworks.

## Features

- Pure NumPy implementation of a three-layer neural network
- Manual backpropagation
- Supports multiple activation functions (ReLU, Sigmoid, Tanh)
- SGD optimizer and learning rate scheduling
- L2 regularization
- Full hyperparameter search functionality
- Modular design: training, testing, hyperparameter tuning, and visualization

## Directory Structure

```
cifar10/
├── model.py                  # Neural network model
├── data.py                   # Data loading and preprocessing
├── trainer.py                # Training loop and visualization
├── hyperparameter_search.py  # Hyperparameter search utilities
├── test.py                   # Model evaluation and visualization
├── cifar.py                  # Main entry point
└── README.md                 # Project documentation
```

## Requirements

- Python 3.6+
- numpy
- matplotlib

## Getting Started

### Dataset

The CIFAR-10 dataset will be automatically downloaded and extracted on first run.

### Training

```bash
# Train with default parameters
python cifar_load_and_test.py --mode train

# Train with custom parameters
python cifar_load_and_test.py --mode train --hidden_size1 200 --hidden_size2 100 --lr 0.01 --batch_size 100 --epochs 50 --reg_lambda 0.01 --activation relu
```

### Hyperparameter Search

```bash
# Search for the best learning rate
python cifar_load_and_test.py --mode search --search_type lr

# Search for the best hidden layer sizes
python cifar_load_and_test.py --mode search --search_type hidden

# Search for the best regularization strength
python cifar_load_and_test.py --mode search --search_type reg

# Search for the best activation function
python cifar_load_and_test.py --mode search --search_type activation

# Full grid search and retrain with the best parameters
python cifar_load_and_test.py --mode search --search_type all
```

### Testing

```bash
# Evaluate a trained model
python cifar_load_and_test.py --mode test --model_path ./cifar10/results/best_model.pkl
```

## Main Hyperparameters

| Argument         | Description                        | Default      |
|------------------|------------------------------------|--------------|
| --hidden_size1   | Size of first hidden layer          | 100          |
| --hidden_size2   | Size of second hidden layer         | 100          |
| --activation     | Activation function (relu/sigmoid/tanh) | relu    |
| --lr             | Learning rate                      | 0.01         |
| --batch_size     | Batch size                         | 100          |
| --epochs         | Number of training epochs           | 300          |
| --reg_lambda     | L2 regularization strength          | 0.01         |
| --lr_decay       | Use learning rate decay             | False        |
| --decay_rate     | Learning rate decay factor          | 0.95         |

## Outputs and Visualization

After training, results and visualizations are saved in `./cifar10/results/`:

- Training and validation loss/accuracy curves
- Visualization of model weights (as images and heatmaps)
- Confusion matrix on the test set
- Random sample predictions

## Model Architecture

- **Input layer:** 3072 neurons (flattened 32x32x3 image)
- **Hidden layer 1:** Configurable size (default 100)
- **Hidden layer 2:** Configurable size (default 100)
- **Output layer:** 10 neurons (10 classes)

## Notes

- The dataset is automatically downloaded on first run; ensure you have a working internet connection.
- Hyperparameter search can be time-consuming; for quick tests, reduce the number of epochs.
- For best results, use the best hyperparameters found by the search for final training.

## Performance

With default hyperparameters and 50 epochs, the model typically achieves 45-50% accuracy on the CIFAR-10 test set. 