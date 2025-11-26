import pandas as pd
import us
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score, explained_variance_score


# CSV File
df = pd.read_csv("/Users/andrea/Desktop/DSCI-510/dsci510_fall2025_final_project/Data/Behavioral_Risk_Factor_Surveillance_System__BRFSS__Prevalence_Data__2011_to_present_.csv")


# Clean and converted data types
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Data_value"] = pd.to_numeric(df["Data_value"], errors="coerce")
df = df.dropna(subset=["Year", "Data_value", "Sample_Size"])

# Filter by year range
START_YEAR, END_YEAR = 2016, 2023
df = df[(df["Year"] >= START_YEAR) & (df["Year"] <= END_YEAR)]

# Kept only U.S. states
states = [s.abbr for s in us.states.STATES]
df = df[df["Locationabbr"].isin(states)]

# Sample due to large dataset
Sample_Size = 25000
df = df.sample(n=Sample_Size, random_state=42)


# Encoded categorical variables
state_encoder = LabelEncoder()
df["State"] = state_encoder.fit_transform(df["Locationabbr"])

resp_encoder = LabelEncoder()
df["Response"] = resp_encoder.fit_transform(df["Response"])

topic_encoder = LabelEncoder()
df["BRFSS_Survey_Questions"] = topic_encoder.fit_transform(df["Topic"])

# Features & target
X = df[["State", "BRFSS_Survey_Questions", "Response", "Year"]]
y = df["Data_value"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)


# Random Forest Regressor Feature Importance
importances = rf_model.feature_importances_
features = X.columns
feature_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=True)

plt.figure(figsize=(8,5))
plt.barh(feature_df["Feature"], feature_df["Importance"], color="teal")
plt.title("Random Forest Regressor - BRFSS Prevalence Data")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Ramdom forest regression - Data Metrics
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
medae = median_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("Median AE:", medae)
print("R^2:", r2)
print("Explained Variance Score:", evs)


# Heatmap: States vs Years
pivot_df = df.pivot_table(
    values="Data_value",
    index="Locationabbr",
    columns="Year"
)
pivot_df = pivot_df.sort_index(ascending=True)

# Plot Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(pivot_df, cmap="viridis", annot=True, fmt=".1f")
plt.title("BRFSS Prevalence Data Heatmap")
plt.xlabel("Year")
plt.ylabel("State")
plt.show()




