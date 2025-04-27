# app.py

from flask import Flask, render_template, request
import pickle

# Load model dan vectorizer
with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Inisialisasi Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        text_input = request.form['text_input']
        
        # Transform input
        text_vector = vectorizer.transform([text_input])
        
        # Predict
        result = model.predict(text_vector)
        
        # Convert output
        if result[0] == 1 or result[0] == "Positive":
            prediction = "Positive"
        else:
            prediction = "Negative"
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
