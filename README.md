# Coupon Acceptance Prediction

## Overview

This project aims to predict whether a customer will accept a promotional coupon based on demographic information, driving conditions, destination, and coupon characteristics.

The project explores multiple machine learning classification algorithms and investigates the impact of feature engineering on model performance. Hyperparameter optimization using Optuna is also implemented to improve the predictive performance of the CatBoost model.

## Dataset

The dataset contains customer demographic information, driving context, and coupon-related attributes.

**Target Variable:**
- `Y`: Indicates whether a customer accepts the offered coupon.
  - `1`: Accepted
  - `0`: Rejected

**Feature Categories:**
- **Customer Demographics:** Gender, age, marital status, education, income, occupation, and presence of children.
- **Driving Context:** Destination, passengers, weather, temperature, time, and distance to the coupon location.
- **Coupon Information:** Coupon type, expiration time, and frequency of visits to relevant establishments.

The dataset consists of separate training and testing files.

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Examining dataset structure and feature distributions.
- Analyzing target class distribution.
- Identifying missing values and duplicate records.
- Investigating correlations between features.
- Evaluating relationships between features and coupon acceptance.

### 2. Data Cleaning
- Handling missing values.
- Removing redundant features based on correlation analysis.
- Investigating highly correlated features to reduce redundancy.

### 3. Feature Engineering

Three additional features are developed to capture customer behavior and contextual information:

- **Coupon Distance (`to_Coupon`):** Combines travel-time indicators (`toCoupon_GEQ15min` and `toCoupon_GEQ25min`) into a categorical feature representing proximity to the coupon location.
- **Coupon Frequency (`coupon_freq`):** Extracts customer visit frequency to the establishment associated with the offered coupon, representing customer preferences.
- **Occupation Classification (`occupation_class`):** Groups occupations into broader categories to simplify categorical representations.

### 4. Data Preprocessing
- Handling missing categorical values using mode imputation.
- Encoding categorical variables using One-Hot Encoding.
- Scaling numerical features using StandardScaler.
- Combining categorical and numerical features into a sparse matrix representation.

Two experimental approaches are implemented:

1. **With Feature Engineering:** Uses the additional engineered features to represent customer behavior and coupon context.
2. **Without Feature Engineering:** Uses the original dataset with standard preprocessing to evaluate model performance without the additional engineered features.

### 5. Model Development

Three classification algorithms are evaluated:

| Model | Description |
|---|---|
| XGBoost | Gradient boosting-based classification |
| Support Vector Classifier (SVC) | Kernel-based classification |
| CatBoost | Gradient boosting algorithm |

Model evaluation uses ROC-AUC as the primary metric. Five-fold cross-validation is applied in the feature-engineered experiments.

### 6. Hyperparameter Optimization

Optuna is used to optimize the CatBoost model through 50 hyperparameter optimization trials.

The optimization process explores:
- Number of iterations
- Tree depth
- Learning rate
- L2 regularization
- Random strength
- Bagging temperature
- Border count

The best hyperparameter configuration is then used to train the final CatBoost model.

## Experimental Results

### 1. Model Performance with Feature Engineering

The following results were obtained using 5-fold cross-validation with ROC-AUC as the evaluation metric.

| Model | ROC-AUC (5-Fold CV) |
|---|---:|
| CatBoost | **0.7957** |
| XGBoost | 0.7796 |
| SVC | 0.7229 |

CatBoost achieved the highest cross-validation ROC-AUC among the three evaluated models.

### 2. Model Performance without Feature Engineering

The following results were obtained from model evaluation on the same training data used for model fitting.

| Model | Training ROC-AUC |
|---|---:|
| XGBoost | **0.9963** |
| CatBoost | 0.9667 |
| SVC | 0.9084 |

XGBoost achieved the highest training ROC-AUC, followed by CatBoost and SVC.

**Note:** These are training-set scores, not cross-validation or independent validation results. Therefore, they should not be directly compared with the five-fold cross-validation results above.

### 3. CatBoost Hyperparameter Optimization

Optuna was used to optimize CatBoost through 50 trials, using an 80:20 training-validation split.

<|-- table --|>

| Evaluation | ROC-AUC |
|---|---:|
| Best Optuna Validation Score | **0.8231** |

**Best Hyperparameters:**

| Parameter | Best Value |
|---|---:|
| Iterations | 345 |
| Depth | 10 |
| Learning Rate | 0.03386 |
| L2 Leaf Regularization | 3.3285 |
| Random Strength | 1.0678 |
| Bagging Temperature | 0.7349 |
| Border Count | 146 |

The optimized CatBoost model achieved a validation ROC-AUC of **0.8231**.

## Conclusion

This project explored different machine learning approaches for predicting customer coupon acceptance, comparing models with and without additional feature engineering.

The main findings are:

1. **CatBoost performed best in the feature-engineered experiments**, achieving a five-fold cross-validation ROC-AUC of 0.7957, outperforming XGBoost and SVC.

2. **The models without feature engineering achieved higher training ROC-AUC scores.** XGBoost obtained 0.9963, while CatBoost achieved 0.9667. However, these scores were calculated on training data and do not establish better generalization performance.

3. **Hyperparameter optimization improved the evaluation process for CatBoost.** Optuna identified a configuration that achieved a validation ROC-AUC of 0.8231 after 50 trials.

Overall, the experiments demonstrate the importance of model selection, feature representation, and hyperparameter optimization in customer coupon acceptance prediction. Further evaluation using consistent cross-validation or an independent validation set is needed to determine the most reliable model for unseen data.

## Technologies & Libraries

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost, CatBoost
- **Hyperparameter Optimization:** Optuna
- **Development Environment:** Google Colab

## Project Structure

```text
Coupon-Prediction/
│
├── Cuopon_Prediction.ipynb
└── README.md
```

## How to Run

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd Coupon-Prediction
   ```

2. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost optuna
   ```

3. Prepare the dataset:
   - Place `train.csv` and `test.csv` in Google Drive.
   - Adjust the dataset paths in the notebook if necessary.

4. Open `Cuopon_Prediction.ipynb` in Google Colab or Jupyter Notebook.

5. Run the notebook cells sequentially to perform data analysis, preprocessing, feature engineering, model training, hyperparameter optimization, and prediction.
