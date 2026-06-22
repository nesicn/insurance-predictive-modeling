# Medical Insurance Predictive Modeling

A procedural Python data science pipeline built to explore data preprocessing and linear regression analysis. This project utilizes Scikit-Learn and Pandas to construct a model that estimates health insurance charges based on individual demographic and lifestyle risk factors.

## Repository Structure

```text
insurance-predictive-modeling/
├── data/
│   └── insurance.csv       # Sample insurance dataset
├── src/
│   └── model.py            # Linear regression pipeline
├── .gitignore              # Safely isolates local .venv tracking
└── requirements.txt        # Third-party project dependencies
```
Handled categorical values by mapping the text column smoker into binary numeric indicators (0 and 1) to make the data mathematically compatible with regression algorithms.

Isolated the data using an 80/20 train-test split configuration to ensure model validation occurs on entirely unseen entries.

Calculated empirical performance markers including Mean Absolute Error (MAE) and the R-squared score to measure accuracy.

## Results

When evaluated against the verification data, the pipeline yielded the following metrics:

Mean Absolute Error (MAE): 4,495.15

R-squared Score: 0.6870

This indicates that approximately 68.7% of the variance in medical insurance charges within this dataset can be directly explained by the four selected attributes: age, BMI, children, and smoking status.
