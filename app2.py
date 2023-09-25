import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style = "darkgrid")


################################################################
###############         Load Dataset            ################ 
################################################################

# Get a list of all the CSV files in the current directory
file_names = glob.glob('data/*.csv')

# Create a list of DataFrames
dfs = []
for file_name in file_names:
    df = pd.read_csv(file_name)
    dfs.append(df)

# Concatenate the DataFrames along the rows axis
df = pd.concat(dfs)
df = df.iloc[:, 1:]

# Drop unused columns
df.drop(columns=['customer.id', 'date.of.report'], inplace=True)

df_loan = df.dropna(subset=['current_ca_credit_limit', 'current_cp_credit_limit'])

# Print the DataFrame
print(df_loan)

################################################################
############### Exploratory Data Analysis (EDA) ################ 
################################################################

# Data summary
df_loan.corr
df_loan.shape
df_loan.describe()
df_loan.info()
df_loan.isnull().sum()
df_loan.columns

# Check target variable
df_loan['default'].value_counts()

# Check all numerical columns
df_loan['default'].hist(figsize = (22, 20))
plt.show()


# Observe correlation plot
fig, ax = plt.subplots( figsize = (22,20) )
corr_matrix = df_loan.corr()
corr_heatmap = sns.heatmap( corr_matrix, annot=True, cmap = "flare", ax=ax, annot_kws={"size": 14})
plt.show()

corrr_heatmap = sns.clustermap(corr_matrix, annot=True, cmap="flare")
plt.show()

# Analysing categorical features
def categorical_valcount_hist(feature):
    print(df_loan[feature].value_counts())
    fig, ax = plt.subplots( figsize = (6,6) )
    sns.countplot(x=feature, ax=ax, data=df_loan)
    plt.show()

categorical_valcount_hist("gender")

categorical_valcount_hist("occupation")

categorical_valcount_hist("nationality")

categorical_valcount_hist("race")

categorical_valcount_hist("marital.status")

categorical_valcount_hist("educational.qualification")

categorical_valcount_hist("type.of.employment")

categorical_valcount_hist("mobile.app.status")

categorical_valcount_hist("opt.out")

categorical_valcount_hist("customer.with.active.credit.card")

categorical_valcount_hist("own.multiple.credit.card")

categorical_valcount_hist("supplementary.credit.card")

categorical_valcount_hist("customer.with.active.financing.product")

categorical_valcount_hist("own.multiple.financing.products")


# Data Analysis

sns.boxplot(x ="default", y="annual.income", showmeans=True, data = df_loan)

sns.boxplot(x ="default", y="age", showmeans=True, data = df_loan)

sns.boxplot(x ="default", y="number.of.dependents", showmeans=True, data = df_loan)

sns.boxplot(x ="default", y="current_ca_credit_limit", showmeans=True, data = df_loan)

sns.boxplot(x ="default", y="current_cp_credit_limit", showmeans=True, data = df_loan)

sns.boxplot(x ="default", y="current.loan.balance", showmeans=True, data = df_loan)
plt.show()

################################################################
###############    Data Cleaning & Preparation  ################ 
################################################################

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# Data Cleaning 
#df = df[df['gender'] != 'Unknown']
#df_loan.dropna(subset=['nationality', 'marital.status','type.of.employment'], inplace=True)

# Separate features and target
X = df_loan.drop('default', axis=1)
y = df_loan['default']

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Define transformers for numerical and categorical columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent category
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

# Create a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Create a preprocessing pipeline
preprocessor_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit and transform the data
X_encoded = preprocessor_pipeline.fit_transform(df_loan)

print(X_encoded) 

################################################################
###############          Model Development      ################ 
################################################################

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Model 1
# Create and train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train_resampled)

# Make predictions on the test set
y_pred = clf.predict(X_test_scaled)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("AUC-ROC Score:", roc_auc_score(y_test, clf.predict_proba(X_test_scaled)[:, 1]))

#Model 2
# Create and train XGBoost classifier
xgb_classifier = xgb.XGBClassifier(
    objective='binary:logistic',  # Binary classification
    random_state=42
)

# Train the model on the training data
xgb_classifier.fit(X_train_scaled, y_train_resampled)

# Make predictions on the test data
y_pred = xgb_classifier.predict(X_test_scaled)

# Calculate AUC-ROC score
roc_auc = roc_auc_score(y_test, xgb_classifier.predict_proba(X_test_scaled)[:, 1])

# Print classification report and AUC-ROC score
print("Classification Report:\n", classification_report(y_test, y_pred))
print("AUC-ROC Score:", roc_auc)


################################################################
###############          Model Evaluation       ################ 
################################################################

from sklearn.metrics import roc_curve, auc

# Run for which classifier you want to check (clf = RF, xgb = SGBoost)
y_prob = clf.predict_proba(X_test_scaled)[:, 1] 
y_prob = xgb_classifier.predict_proba(X_test_scaled)[:, 1] 

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculate the AUC-ROC score
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
