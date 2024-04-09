import streamlit as st 
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

from functions.preprocess import clean_sent

model1 = load_model('./models/accepted1.keras')
model2 = load_model('./models/accepted2.keras')
model3 = load_model('./models/accepted3.keras')


print("Loading label encoder...")
label_encoder = joblib.load('./pkl/label_encoder.pkl')
print("Loading tokenizer...")
tokenizer = joblib.load('./pkl/tokenizer.pkl')

st.set_page_config(page_title='Streamlit APP')

st.title('Email Categorization Model')

subject = st.text_area('Enter subject here: ')
message = st.text_area('Enter Message here: ')

email = subject + ' ' + message

clean_text = clean_sent(email)

tokenized_text = tokenizer.texts_to_sequences([clean_text])
padded_text = pad_sequences(tokenized_text, maxlen=150)

if st.button('Predict'):
    print("Predicting...")
    
    prediction3 = model3.predict(padded_text)
    predicted_class_index3 = prediction3.argmax(axis=1)[0] 
    predicted_class3 = label_encoder.inverse_transform([predicted_class_index3])[0]
    
    st.write('The predicted category is:', predicted_class3)
    st.write('Class probabilities:', prediction3[0][predicted_class_index3])
