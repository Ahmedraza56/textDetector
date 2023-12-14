import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load pre-trained DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

def check_text(text):
    # Tokenize and convert to model input format
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    # Make a prediction
    outputs = model(**inputs)

    # Get predicted label
    prediction = torch.argmax(outputs.logits).item()

    # Analyze the prediction and classify as AI-generated or human-written
    if prediction == 0:  # You may need to adjust this based on your model
        return "This text is likely human-written."
    else:
        return "This text appears to be AI-generated."

def main():
    st.title("Text Detector")

    # Get user input
    user_input = st.text_area("Enter text:")

    if st.button("Check"):
        if user_input:
            result = check_text(user_input)
            st.write(result)
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
