from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model/complaint_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

complaints = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit():

    text = request.form["complaint"]

    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]

    complaints.append({
        "text": text,
        "priority": prediction
    })

    return render_template("dashboard.html", complaints=complaints)

if __name__ == "__main__":
    app.run(debug=True)