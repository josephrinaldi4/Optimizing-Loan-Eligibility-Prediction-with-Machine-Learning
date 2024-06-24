```markdown
# Loan Eligibility Prediction with Machine Learning

## Overview
This project focuses on developing machine learning models to predict loan eligibility based on historical data. The goal is to automate the loan approval process for a financial company, improving efficiency and reducing potential losses.

## Project Structure
The project is organized as follows:

- **data/**: Contains the dataset used for training and testing the models.
- **notebooks/**: Jupyter notebooks with exploratory data analysis, data preprocessing, and model building.
- **scripts/**: Python scripts for data preprocessing, model training, and evaluation.
- **results/**: Outputs such as model performance metrics and visualizations.

## Data Dictionary
The dataset includes the following attributes:

- **Loan_ID**: Unique Loan ID
- **Gender**: Gender of the applicant (Male/Female)
- **Married**: Marital status of the applicant (Yes/No)
- **Dependents**: Number of dependents of the applicant
- **Education**: Education level of the applicant (Graduate/Not Graduate)
- **Self_Employed**: Whether the applicant is self-employed (Yes/No)
- **ApplicantIncome**: Income of the applicant ($)
- **CoapplicantIncome**: Income of the co-applicant ($)
- **LoanAmount**: Loan amount (in thousands of dollars)
- **Loan_Amount_Term**: Term of loan (in months)
- **Credit_History**: Credit history (meets guidelines or not)
- **Property_Area**: Property area type (Urban/Semi-Urban/Rural)
- **Loan_Status**: Loan approval status (1 - Yes, 0 - No)

## Installation
To run this project locally, you'll need Python and the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

Install the necessary libraries using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

## Usage
### Data Preprocessing
The data preprocessing steps include:
- Handling missing values
- Encoding categorical variables
- Normalizing numerical features

### Model Building
Various machine learning models are explored and evaluated, including:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)

### Evaluation
Model performance is evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Running the Notebooks
Execute the Jupyter notebooks in the `notebooks/` directory to follow the data preprocessing, model building, and evaluation steps.

### Running the Scripts
Use the scripts in the `scripts/` directory to preprocess data, train models, and evaluate performance.

## Results
The results of the model evaluations are saved in the `results/` directory, including performance metrics and visualizations of feature importances.

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests with your improvements.

## License
This project is licensed under the MIT License.

## Contact
For any inquiries, please contact josephrinaldi4@gmail.com.

---

Thank you for checking out this project! We hope it helps you understand and implement loan eligibility prediction using machine learning.
```
