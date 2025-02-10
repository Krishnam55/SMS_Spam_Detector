import nltk
import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = [i for i in text if i.isalnum()]
    
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    
    y = [ps.stem(i) for i in y]
    
    return " ".join(y)

import joblib
tk = joblib.load("optimized_tfidf_vectorizer.pkl")  
model = joblib.load("optimized_spam_classifier.pkl")  

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

# Custom CSS for curved grey border
st.markdown("""
   <style>
   .curved-border {
       border: 2px solid #808080;
       border-radius: 15px;
       padding: 20px;
       min-height: 350px;
       width: 800px;
       position: absolute;
       left: 50%;
       transform: translateX(-50%);
       display: flex;
       flex-direction: column;
       justify-content: space-between;
   }
   .author-attribution {
       text-align: right;
       color: grey;
       position: absolute;
       bottom: 10px;
       right: 20px;
   }
   </style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='curved-border'>",
    unsafe_allow_html=True
)
st.title("SMS Spam Detection Model")
input_sms = st.text_area("📩 Enter the SMS below:", height=100)

if st.button('🚀 Predict'):
    with st.spinner('Analyzing...'):
        transformed_sms = transform_text(input_sms)
        vector_input = tk.transform([transformed_sms])
        result = model.predict(vector_input)[0]

st.markdown("<p class='author-attribution'>By Pakalapati S R S Krishnam Raju</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)


if 'result' in locals():
    if result == 1:
        st.error("🔥 This message is Spam!")
    else:
        st.success("✅ This message is Not Spam!")
    st.markdown("""
        <style>
       .curved-border {
           min-height: 450px;
       </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <br>
    <p style='text-align:center; color:grey; display: block; padding-top: 3px;'>
        Under the Guidance of Abdul Aziz Md
    </p>
""", unsafe_allow_html=True)


