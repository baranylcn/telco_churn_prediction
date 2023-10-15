# TELCO CHURN PREDICTION

![image](https://github.com/baranylcn/telco_churn_prediction/assets/98966968/0739ab58-bbc1-4050-a6b8-1cf666be2a25)


## Business Problem
A machine learning model that can predict customers leaving the company development is expected.

## Dataset Story
Telco churn data revealed fictitious home phone and Internet services to 7,043 California customers in the third quarter.
Contains information about a telecom company. Which customers have left, stayed or signed up for their service shows.

### CSV File
- Total Features : 21
- Total Row : 7043
- CSV File Size : 977.5 KB



| Feature | Description |
|----------|----------|
| CustomerId  | Customer's ID  |
| Gender  | Gender  |
| SeniorCitizen  | Whether the customer is old (1, 0) |
| Partner  | Whether the customer has a partner (Yes, No ) |
| Dependents  | Whether the customer is dependent on (Yes, No)  |
| tenure  | Number of months the customer stayed in the company  |
| PhoneService  | Whether the customer has a phone service (Yes, No)  |
| MultipleLines  | Whether the customer has more than one line (Yes, No, No phone service)  |
| InternetService  | Customer's internet service provider (DSL, Fiber Optics, No) |
| OnlineSecurity  | Whether the customer has online security (Yes, No, no Internet service)  |
| OnlineBackup  | Whether the customer has an online backup (Yes, No, no Internet service) |
| DeviceProtection  | Whether the customer has device protection (Yes, No, no Internet service)  |
| TechSupport  | Whether the customer receives technical support (Yes, No, no Internet service) |
| StreamingTV  | Whether the customer has a TV broadcast (Yes, No, no internet service  |
| StreamingMovies  | Whether the customer has a movie stream (Yes, No, no Internet service) |
| Contract  | Customer's contract duration ( Month to month, One year, Two years) |
| PaperlessBilling  | Whether the customer has a paperless invoice (Yes, No) |
| PaymentMethod  | Customer's payment method (Electronic check, Post check, Bank transfer (automatic), Credit card (automatic)) |
| MonthlyCharges  | Amount collected monthly from the customer |
| TotalCharges  | Total amount collected from the customer |
| Churn  | Whether the customer is using (Yes or No) |


## STEPS
- Checking DataFrame
- Categorical and numerical analysis
- Checking correlation
- Missing Values
- Outliers
- Encoding
- Scaling
- Base Model
- Generating new features
- LightGBM
- Hyperparameter Optimization
- Importance levels of variables
- Prediction example
