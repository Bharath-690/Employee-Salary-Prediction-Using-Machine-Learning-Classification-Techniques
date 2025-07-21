import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE
import os

# Load data
df = pd.read_csv("adult 3.csv")

# Basic cleanup
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

# Filter categories
allowed_education = ['Bachelors', 'HS-grad', 'Some-college', 'Masters', 'Assoc', 'Doctorate']
df = df[df['education'].isin(allowed_education)]
df = df[df['age'] <= 70]
df = df[~df['workclass'].isin(['Without-pay', 'Never-worked'])]
df = df[~df['marital-status'].isin(['Married-AF-spouse'])]
df = df[~df['occupation'].isin(['Armed-Forces'])]

# Outlier removal using IQR
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

for col in ['age', 'hours-per-week']:
    df = remove_outliers(df, col)

# Create "experience" as synthetic column (optional)
df['experience'] = df['age'] - 18
df['experience'] = df['experience'].apply(lambda x: max(x, 0))

# Final features to be used
features = ['age', 'education', 'occupation', 'hours-per-week', 'experience']
target = 'income'

# Encode categorical columns
label_encoders = {}
for col in ['education', 'occupation']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target separately
target_le = LabelEncoder()
df[target] = target_le.fit_transform(df[target])
label_encoders['income'] = target_le

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Handle imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train_res)


# --- AutoML pipeline function added here ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, mean_squared_error, r2_score

def auto_ml_pipeline(df):
    # Your AutoML pipeline function (full version from earlier)
    ...

def auto_ml_pipeline(df):
    report = {}
    visuals = []

    # 1. Data Cleaning
    df = df.drop_duplicates()
    df = df.dropna(thresh=0.7 * len(df), axis=1)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    # 2. Encoding
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # 3. Correlation heatmap
    fig1, ax = plt.subplots()
    sns.heatmap(df.corr(), ax=ax, cmap='coolwarm', annot=False)
    visuals.append(fig1)

    # 4. Train/test split
    target_col = df.columns[-1]
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    problem_type = 'classification' if y.nunique() <= 10 and y.dtype in [int, bool] else 'regression'
    report["Problem Type"] = problem_type

    # 5. Model selection
    scores = {}

    if problem_type == 'classification':
        models = {
            "RandomForest": RandomForestClassifier(),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "GradientBoosting": GradientBoostingClassifier()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            f1 = f1_score(y_test, preds, average='weighted')
            scores[name] = f1

        best_model_name = max(scores, key=scores.get)
        best_model = models[best_model_name]
        preds = best_model.predict(X_test)

        report["Best Model"] = best_model_name
        report["F1 Score"] = scores[best_model_name]
        report["Classification Report"] = classification_report(y_test, preds, output_dict=True)

        # Confusion Matrix
        fig2, ax2 = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', ax=ax2)
        ax2.set_title(f"Confusion Matrix - {best_model_name}")
        visuals.append(fig2)

    else:
        models = {
            "RandomForest": RandomForestRegressor(),
            "LinearRegression": LinearRegression(),
            "GradientBoosting": GradientBoostingRegressor()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = mean_squared_error(y_test, preds, squared=False)
            scores[name] = -rmse  # Lower RMSE = better, so we invert for max()

        best_model_name = max(scores, key=scores.get)
        best_model = models[best_model_name]
        preds = best_model.predict(X_test)

        report["Best Model"] = best_model_name
        report["RMSE"] = mean_squared_error(y_test, preds, squared=False)
        report["R2 Score"] = r2_score(y_test, preds)

        # Residuals Plot
        fig3, ax3 = plt.subplots()
        sns.residplot(x=preds, y=y_test - preds, ax=ax3, lowess=True, line_kws={'color': 'red'})
        ax3.set_title(f"Residual Plot - {best_model_name}")
        visuals.append(fig3)

    return report, visuals
