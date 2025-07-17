import streamlit as st
from transformers import pipeline

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="YouTube Toxic Comment Classifier",
    page_icon="🛡️",
    layout="centered",
)

# -------------------------------
# Custom CSS for dark theme and styles
# -------------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: #ffffff;
    }
    .main {
        background-color: #1e1e1e;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.5);
    }
    .title {
        color: #bb86fc;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
    }
    .subtitle {
        color: #03dac6;
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 2rem;
    }
    .prediction {
        font-size: 1.2em;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .toxic {
        background-color: #cf6679;
        color: #000000;
    }
    .nontoxic {
        background-color: #03dac6;
        color: #000000;
    }
    textarea {
        background-color: #333333 !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Toxic keywords list
# -------------------------------
toxic_keywords = [
    'fuck', 'fucking', 'fucked', 'shit', 'shitty', 'bullshit', 'bitch', 'bitches',
    'ass', 'asshole', 'motherfucker', 'mf', 'cunt', 'dick', 'dicks', 'cock',
    'pussy', 'faggot', 'fag', 'dyke', 'tranny', 'nigger', 'nigga', 'chink', 'spic', 'kike',
    'slut', 'whore', 'bastard', 'retard', 'retarded', 'moron', 'idiot', 'stupid', 'dumb',
    'loser', 'worthless', 'pathetic', 'disgusting', 'fat', 'ugly', 'kill yourself', 'kys',
    'die', 'drop dead', 'kill', 'burn in hell', 'go to hell', 'hate you', 'hell'
]

# -------------------------------
# Load the toxic comment classification pipeline
# -------------------------------
@st.cache_resource
def load_pipeline():
    classifier_pipeline = pipeline(
        "text-classification",
        model="toxic_classifier_model",
        tokenizer="toxic_classifier_model",
        framework="tf"
    )
    return classifier_pipeline

classifier = load_pipeline()

# -------------------------------
# Streamlit app layout
# -------------------------------
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">🛡️ YouTube Toxic Comment Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Classify comments as toxic or non-toxic instantly.</div>', unsafe_allow_html=True)

# Text input
user_input = st.text_area("💬 Enter your comment below:", height=100)

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a comment for prediction.")
    else:
        comment_lower = user_input.lower()
        is_toxic_keyword = any(word in comment_lower for word in toxic_keywords)

        if is_toxic_keyword:
            st.markdown(f"<div class='prediction toxic'>🏷️ Prediction: <strong>TOXIC</strong><br>🔢 Confidence: Keyword detected</div>", unsafe_allow_html=True)
        else:
            result = classifier(user_input)[0]
            label = 'TOXIC' if result['label'] == 'LABEL_1' else 'NON-TOXIC'
            confidence = round(result['score'] * 100, 2)

            if label == "TOXIC":
                st.markdown(f"<div class='prediction toxic'>🏷️ Prediction: <strong>{label}</strong><br>🔢 Confidence: {confidence}%</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='prediction nontoxic'>🏷️ Prediction: <strong>{label}</strong><br>🔢 Confidence: {confidence}%</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown(
    """
    <hr>
    <div style='text-align: center; font-size: 0.9em; color: gray;'>
        Developed by Faizan | Powered by BERT & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)












