# SynthFix: 

This project trains a model to improve code quality and fix bugs using reinforcement learning techniques like Proximal Policy Optimization (PPO) combined with Supervised Fine-Tuning (SFT). The model uses both structural and quality-based rewards to evaluate and enhance code generation.

## Installation

Ensure you have Python 3.10 installed. Install the required packages with:

```bash
pip install -r requirements.txt
```

## Usage

1. Prepare Data: Place your buggy and fixed code data in the input directory.
2. Run Training: Execute the `main.ipynb` notebook to start training the model. The training includes both SFT and PPO phases.
3. Generate Predictions: Use the provided functions in the notebook to generate predictions and evaluate them with CodeBLEU.

## Key Steps in `main.ipynb`

1. Import Libraries: Import necessary libraries and packages.
2. Initialize Tokenizer and Device: Set up the tokenizer and the device (CPU/GPU) for training.
3. Load and Preprocess Data: Load buggy and fixed code data, shuffle, and preprocess it.
4. Split Data: Split the data into training, validation, and test sets.
5. Process Data: Convert the data into model inputs.
6. Calculate Rewards: Define functions to calculate code quality and CFG-based rewards.
7. Define Critic Model: Create a neural network model for the critic.
8. Train Model: Train the model using both SFT and PPO methods.
9. Generate Predictions: Generate and evaluate code predictions using the trained model.

## Evaluation

Evaluate the generated code using the CodeBLEU metric to measure the quality and correctness of the fixes.
