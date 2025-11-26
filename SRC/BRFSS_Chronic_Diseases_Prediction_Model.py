# Chronic Disease - Prediction Model (ML)

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


# CSV File
df = pd.read_csv("/Users/andrea/Desktop/DSCI-510/dsci510_fall2025_final_project/Data/BRFSS2023.csv")

# Random Sampling
sample_size = 25000
if len(df) > sample_size:
    df = df.sample(n=sample_size, random_state=42)

# Target variable
df = df[df['genhlth'].isin([1, 2, 3, 4, 5])]
y = df['genhlth']

# Features
features = [
    'physhlth', 'menthlth', 'poorhlth',  # numeric health features
    'smoke100', 'alcday4', 'exerany2',
    '_bmi5',
    'diabete4', 'cvdinfr4', 'asthma3',
    '_age_g', 'educa', 'income3', '_sex',
    'colncncr', 'chccopd3'
]

df = df[features + ['genhlth']]
X = df[features]

# Preprocessing
numeric = ['physhlth', 'menthlth', 'poorhlth', '_bmi5']
categorical = [
    'smoke100', 'alcday4', 'exerany2',
    'diabete4', 'cvdinfr4', 'asthma3',
    '_age_g', 'educa', 'income3', '_sex',
    'colncncr', 'chccopd3'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical)
    ]
)

# Random Forest Classifiier Model - pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1))
])

# Train-test split - RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Classification report
print(classification_report(y_test, y_pred))

# Macro F1
macro_f1 = f1_score(y_test, y_pred, average='macro')
print("Macro F1 Score:", round(macro_f1, 3))

# after model training
importances = model.named_steps['classifier'].feature_importances_
cat_feature_names = model.named_steps['preprocessor'] \
                       .named_transformers_['cat'] \
                       .named_steps['encoder'] \
                       .get_feature_names_out(categorical)

# Label maps for numeric and categorical labels
numeric_labels = {
    'physhlth': 'Physical Health',
    'menthlth': 'Mental Health',
    'poorhlth': 'Poor Health',
    '_bmi5': 'BMI'
}
simple_labels = {
    'diabete4': 'Diabetes',
    'cvdinfr4': 'Heart Attack',
    'asthma3': 'Asthma',
    'colncncr': 'Colon Cancer',
    'chccopd3': 'COPD',
    'smoke100': 'Smoking',
    'alcday4': 'Alcohol',
    'exerany2': 'Exercise',
    '_age_g': 'Age',
    'educa': 'Education',
    'income3': 'Income',
    '_sex': 'Gender'
}

grouped_importance = {}

# Numeric labels
for i, feat in enumerate(numeric):
    name = numeric_labels.get(feat, feat)
    grouped_importance[name] = grouped_importance.get(name, 0) + importances[i]

# Categorical labels (aggregated)
offset = len(numeric)
for j, feat in enumerate(cat_feature_names):
    base_name, _ = feat.rsplit('_', 1)  # split from right
    name = simple_labels.get(base_name, base_name)
    grouped_importance[name] = grouped_importance.get(name, 0) + importances[offset + j]

# DataFrame built
feat_df = pd.DataFrame({
    'feature': list(grouped_importance.keys()),
    'importance': list(grouped_importance.values())
}).sort_values('importance', ascending=False)
print(feat_df)  #check that 'Age' is there

# Random Forest Classifier Plot
plt.figure(figsize=(10, 8))
plt.barh(feat_df['feature'][::-1], feat_df['importance'][::-1], color='skyblue')
plt.xlabel('Importance')
plt.title("Random Forest -2023 BRFSS")
plt.subplots_adjust(left=0.3)
plt.tight_layout()
plt.show()


# Confusion Matrix - General Health
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Greens", normalize="true")
plt.title('Confusion Matrix - General Health')
plt.show()

# Actual vs Predicted Distribution
df_plot = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
plt.figure(figsize=(8,5))
sns.countplot(data=df_plot, x='Actual', color='blue', label='Actual', alpha=0.6)
sns.countplot(data=df_plot, x='Predicted', color='red', label='Predicted', alpha=0.4)
plt.legend()
plt.xlabel('General Health (1=Excellent, 5=Poor)')
plt.title("2023 BRFSS Actual vs Predicted General Health")
plt.show()
