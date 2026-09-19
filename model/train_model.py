import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

data = pd.read_csv("../dataset/complaints.csv")

X = data["complaint"]
y = data["priority"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

pickle.dump(model, open("../model/complaint_model.pkl", "wb"))
pickle.dump(vectorizer, open("../model/vectorizer.pkl", "wb"))

print("Model trained successfully")