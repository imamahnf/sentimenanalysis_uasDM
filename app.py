import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_path = 'sentiment_analysis_model.pkl'
model = joblib.load(model_path)

# Define prediction function
def predict_sentiment(text):
    prediction = model.predict([text])[0]
    return prediction

# Streamlit App Setup
st.title("Sentiment Analysis App")
st.write("Upload a CSV file or enter text to analyze sentiment.")

# Text input section
st.header("Analyze Text")
user_text = st.text_area("Enter text for sentiment analysis:")
if st.button("Analyze Text"):
    if user_text.strip():
        result = predict_sentiment(user_text)
        st.success(f"Predicted Sentiment: {result}")
    else:
        st.error("Please enter some text.")

# File upload section
st.header("Upload CSV File")
file = st.file_uploader("Upload a CSV file with a 'text' column:", type=['csv'])

if file is not None:
    try:
        data = pd.read_csv(file)
        if 'text' in data.columns:
            data['predicted_sentiment'] = data['text'].apply(predict_sentiment)
            st.success("File successfully processed!")
            st.dataframe(data[['text', 'predicted_sentiment']])
            # Option to download processed file
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name='sentiment_analysis_results.csv',
                mime='text/csv'
            )
        else:
            st.error("CSV file must contain a 'text' column.")
    except Exception as e:
        st.error(f"Error processing file: {e}")

st.info("Developed by Your Name - Sentiment Analysis Model Deployment")
