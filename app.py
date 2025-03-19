import streamlit as st
import nltk
import joblib
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Ensure nltk data path
nltk.data.path.append('/Users/krishnam/nltk_data')
nltk.download('punkt')
nltk.download('stopwords')

# PorterStemmer initialization
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load models
tk = joblib.load("optimized_tfidf_vectorizer.pkl")  
model = joblib.load("optimized_spam_classifier.pkl")  

st.set_page_config(page_title="SMS Spam Detector", page_icon="🕵️‍♂️📩", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #e0f7fa, #ffffff);
    }
    .custom-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #0277bd;
        margin-bottom: 30px;
    }
    .custom-container {
        background-color: transparent !important; /* Transparent background */
        border: none !important;                 /* Removes the border */
        padding: 0 !important;                   /* Ensures no extra spacing */
        box-shadow: none !important;             /* Removes any shadow */
    }

    .sms-label {
        color: #0277bd !important;
        font-weight: bold !important;
        margin-bottom: 5px !important;
        display: block !important; 
    }
    .stTextArea textarea {
        background-color: #ffffff !important;
        border: 2px solid #4fc3f7 !important;
        border-radius: 10px !important;
        color: #0277bd !important;
        caret-color: #0277bd !important;
        margin-top: -10px !important;
    }
    .stButton>button {
        background-color: #ffffff !important;
        color: #0277bd !important;
        border: 2px solid #4fc3f7 !important;
        border-radius: 10px !important;
        padding: 8px 16px !important;
        font-weight: bold !important;
        cursor: pointer !important;
    }
    .stButton>button:hover {
        background-color: #e1f5fe !important;
    }
    .result-box {
        text-align: center;
        padding: 15px;
        border-radius: 12px;
        margin-top: 20px;
        font-weight: bold;
    }
    .spam {
        background-color: #f44336;
        color: #ffffff;
    }
    .not-spam {
        background-color: #4CAF50;
        color: #ffffff;
    }
    .author-attribution {
        text-align: center;
        color: #0277bd;
        font-weight: bold;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='custom-title'>📩 SMS Spam Detector</div>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
    st.markdown("<div class='sms-label'>Enter the SMS below:</div>", unsafe_allow_html=True)
    input_sms = st.text_area("", height=120, key='custom-textarea')

    if st.button('🚀 Predict'):
        with st.spinner('Analyzing...'):
            transformed_sms = transform_text(input_sms)
            vector_input = tk.transform([transformed_sms])
            result = model.predict(vector_input)[0]

        if result == 1:
            st.markdown("<div class='result-box spam'>🔥 This message is Spam!</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-box not-spam'>✅ This message is Not Spam!</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='author-attribution'>By Pakalapati S R S Krishnam Raju</div>", unsafe_allow_html=True)
