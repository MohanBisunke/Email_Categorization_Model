import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
from functions.preprocess import clean_sent
 

model = load_model('./models/accepted3.keras')
label_encoder = joblib.load('./pkl/label_encoder.pkl')
tokenizer = joblib.load('./pkl/tokenizer.pkl')

def predict_email(subject, message):
    try:
        email = subject + ' ' + message
        clean_text = clean_sent(email)

        tokenized_text = tokenizer.texts_to_sequences([clean_text])
        padded_text = pad_sequences(tokenized_text, maxlen=200)

        prediction = model.predict(padded_text)
        predicted_class_index = prediction.argmax(axis=1)[0]
        predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
        prediction_probability = prediction[0][predicted_class_index]

        return {
            'prediction': prediction.tolist(),
            'predicted_class': predicted_class,
            'prediction_probability': float(prediction_probability),
        }
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return {'error': 'Prediction Error'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict using command line arguments')
    parser.add_argument('--subject', type=str, help='Email subject', required=True)
    parser.add_argument('--message', type=str, help='Email message', required=True)
    args = parser.parse_args()

    
    prediction_result = predict_email(args.subject, args.message)
    prediction_value = prediction_result['prediction']
    predicted_class_value = prediction_result['predicted_class']
    prediction_probability_value = prediction_result['prediction_probability']

    
    print('Prediction:', prediction_value)
    print('Predicted Class:', predicted_class_value)
    print('Prediction Probability:', prediction_probability_value)
