import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset (you can replace with real dataset)
data = pd.read_csv("crop_data.csv")

X = data.drop("label", axis=1)
y = data["label"]

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open("crop_model.pkl", "wb"))

print("Model Saved Successfully")
