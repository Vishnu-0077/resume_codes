# Diabetics Neural Network - Logic and Workflow

This project implements a neural network for predicting diabetes using patient data, built and trained in a Jupyter notebook. The focus here is on the logic and workflow of the code, from data preparation to model evaluation.

## Core Logic and How the Code Works

### 1. **Data Loading and Preprocessing**
- **Data Import:** The dataset (CSV file) is loaded using pandas. The file path must be updated to your local data location.
- **Feature/Label Separation:** Features (`X`) are separated from the target label (`y`). The target column is mapped to binary values (1 for positive, 0 for negative).
- **Train-Test Split:** The data is split into training and testing sets (typically 80/20 split) using `train_test_split` from scikit-learn.
- **Feature Scaling:** Features are standardized using `StandardScaler` to improve neural network training.

### 2. **Model Building**
- **Neural Network Architecture:**
  - Built using Keras Sequential API.
  - Input layer matches the number of features.
  - Two hidden Dense layers with ReLU activation and L2 regularization, each followed by Dropout for regularization.
  - Output layer with sigmoid activation for binary classification (diabetes positive/negative).

#### Example Model Structure:
- Dense (128, relu, L2) → Dropout (0.2) → Dense (64, relu, L2) → Dropout (0.2) → Dense (1, sigmoid)

### 3. **Model Compilation and Training**
- **Compilation:**
  - Loss function: `binary_crossentropy` (for binary classification)
  - Optimizer: Adam
  - Metric: Accuracy
- **Training:**
  - The model is trained for 100 epochs with a batch size of 32.
  - Validation is performed on the test set each epoch to monitor overfitting and performance.

### 4. **Model Evaluation and Comparison**
- **Prediction:**
  - The trained model predicts probabilities on the test set, which are thresholded at 0.5 to get binary predictions.
  - Accuracy is calculated using `accuracy_score` from scikit-learn.
- **Baseline Comparison:**
  - A logistic regression model is also trained and evaluated on the same data for baseline comparison.

## How to Run the Notebook

1. **Install Dependencies:**
   - Python 3.x
   - pandas
   - numpy
   - scikit-learn
   - tensorflow / keras

   Install with pip:
   ```bash
   pip install pandas numpy scikit-learn tensorflow
   ```

2. **Prepare the Dataset:**
   - Place your diabetes dataset CSV (e.g., `diabetesData.csv`) in an accessible location.
   - Update the file path in the notebook to point to your dataset.

3. **Run the Notebook:**
   - Open `neural network of diabetics.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute the cells sequentially to:
     - Load and preprocess data
     - Build, train, and evaluate the neural network
     - Compare with logistic regression baseline

---

**Note:**
- The notebook is designed for educational and experimental purposes. For production use, consider further improvements in data cleaning, model tuning, and error handling.
- GPU acceleration is recommended for faster training.
