from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


from functions.preprocess import clean_sent




app = Flask(__name__)

from flask_cors import CORS

CORS(app)



model = load_model('./models/accepted3.keras')
label_encoder = joblib.load('./pkl/label_encoder.pkl')
tokenizer = joblib.load('./pkl/tokenizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if 'subject' not in data or 'message' not in data:
            return jsonify({'error': 'Invalid input. Provide "subject" and "message".'})

        subject = data['subject']
        message = data['message']
        
        
        email = subject + ' ' + message

        clean_text = clean_sent(email)

        tokenized_text = tokenizer.texts_to_sequences([clean_text])
        padded_text = pad_sequences(tokenized_text, maxlen=200)

        prediction = model.predict(padded_text)
        predicted_class_index = prediction.argmax(axis=1)[0]
        predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
        prediction_probability = prediction[0][predicted_class_index]

        return jsonify({
            'prediction':prediction,
            'predicted_class': predicted_class,
            'prediction_probability': prediction_probability,
            })
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'Internal Server Error'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)