import json
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns


# API Key
KEY_FILE = "API.txt"
with open(KEY_FILE, "r") as f:
    API_KEY = f.readline().strip()

url = "https://api.americashealthrankings.org/graphql"
headers = {
    "Content-Type": "application/json",
    "X-Api-Key": API_KEY
}

# API - GraphQL Query
query = """
query GetMeasureData {
  m1: measure_A(metricId: 16465) {
    name
    description
    data { dateLabel rank state value }
  }
  m2: measure_A(metricId: 16535) {
    name
    description
    data { dateLabel rank state value }
  }
  m3: measure_A(metricId: 17679) {
    name
    description
    data { dateLabel rank state value }
  }
  m4: measure_A(metricId: 16540) {
    name
    description
    data { dateLabel rank state value }
  }
}
"""

# Fetch Data
response = requests.post(url, headers=headers, json={"query": query})

# Checking for HTTP errors
if response.status_code == 200:
    json_response = response.json()
    print(json.dumps(json_response, indent=2)) # pretty print json
else:
    print(f"Request failed with status code {response.status_code}")
    print(response.text)

# storing response
json_response = response.json()
data = json_response["data"]

# Data cleaning to remove NULL from measures
measures = ["m1", "m2", "m3", "m4"]
cleaned = {}

for m in measures:
    df = pd.DataFrame(data[m]["data"])

    df_clean = df[
        (df["state"] !="ALL") &
        (df["rank"].notnull()) &
        (df["value"].notnull())
    ]
    cleaned[m] = df_clean

print("cleaned m1:")
print(cleaned["m1"])
print("cleaned m2:")
print(cleaned["m2"])
print("cleaned m3:")
print(cleaned["m3"])
print("cleaned m4:")
print(cleaned["m4"])

# Combining metric_ids into one table
df = (
    cleaned["m1"][["state", "value"]].rename(columns={"value": "m1"})
    .merge(cleaned["m2"][["state", "value"]].rename(columns={"value": "m2"}), on="state")
    .merge(cleaned["m3"][["state", "value"]].rename(columns={"value": "m3"}), on="state")
    .merge(cleaned["m4"][["state", "value"]].rename(columns={"value": "m4"}), on="state")
)
print(df)

# Ranking ML - Forest Model
X = df[["m1", "m2", "m3"]] # Risk factors(annual), behaviors, & behavioral health - 50 states
y = df["m4"] # overall health - 50 states

feature_names_map = {
    "m1": "Risk Behaviors (Annual)",
    "m2": "Behaviors",
    "m3": "Behavioral Health"
}

#Fiting model (X,y)
model = RandomForestRegressor()
model.fit(X, y)

# Predict scores
df["predict_score"] = model.predict(X)

# Ranking states (higher score = better)
df["predict_rank"] = df["predict_score"].rank(ascending=False)

# Sort by ranking
df_ranked = df.sort_values("predict_rank")
print(df_ranked)

# Plot Ranking ML - Forest Model
importances = model.feature_importances_
feature_names = [feature_names_map[f] for f in X.columns]

plt.figure(figsize=(7,5))
sns.barplot(x=importances, y=feature_names)
plt.title("Health Rankings in 50 States")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Plot - Ranking bar chart (Actual vs Predicted) in 50 States
df_alpha = df.sort_values("state").reset_index(drop=True) # states sorted alphabetically

plt.figure(figsize=(14,6))
plt.bar(df_alpha["state"], df_alpha["predict_rank"], alpha=0.6, label="Predicted Rank") # blue bar = predicted
plt.scatter(df_alpha["state"], df_alpha["m4"].rank(ascending=False), label="Actual Rank") # red dots = actual
plt.xticks(rotation=90, ha="right")
plt.ylabel("Rankings")
plt.title("Predicted vs Actual Health Rankings in 50 States")
plt.legend()
plt.tight_layout()
plt.show()












