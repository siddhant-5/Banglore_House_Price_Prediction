from flask import Flask, request, render_template
import pickle
import numpy as np
import json

# Initialize Flask app
app = Flask(__name__)

# Load model
try:
    with open('banglore_hpp_v2.pickle', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Load columns
try:
    with open("columns.json", "r") as f:
        data_columns = json.load(f)['data_columns']
    print("Columns loaded successfully")
except Exception as e:
    print(f"Error loading columns: {str(e)}")
    data_columns = []

@app.route('/')
def home():
    if not data_columns:
        return "Error: Data columns not loaded", 500
    locations = [col for col in data_columns if col not in ['total_sqft', 'bath', 'price', 'bhk']]
    return render_template('index.html', columns=locations)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or not data_columns:
        locations = [col for col in data_columns if col not in ['total_sqft', 'bath', 'price', 'bhk']]
        return render_template('index.html', 
                               prediction_text='Model not loaded properly',
                               columns=locations)
    
    try:
        location = request.form['location'].lower()
        sqft = float(request.form['sqft'])
        bath = float(request.form['bath'])
        bhk = float(request.form['bhk'])
        
        # Create input array
        x = np.zeros(len(data_columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk
        
        # Set location if it exists in columns
        if location in data_columns:
            x[data_columns.index(location)] = 1

        # Make prediction
        prediction = model.predict([x])[0]
        output = round(prediction, 2)
        
        locations = [col for col in data_columns if col not in ['total_sqft', 'bath', 'price', 'bhk']]
        return render_template('index.html',
                               prediction_text=f'Estimated Price: ₹{output} Lakhs',
                               columns=locations)

    except Exception as e:
        locations = [col for col in data_columns if col not in ['total_sqft', 'bath', 'price', 'bhk']]
        return render_template('index.html',
                               prediction_text=f'Error: {str(e)}',
                               columns=locations)

if __name__ == "__main__":
    app.run(debug=True)