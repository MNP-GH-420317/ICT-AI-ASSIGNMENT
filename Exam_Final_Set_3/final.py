import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F

model = DistilBertForSequenceClassification.from_pretrained("sentiment_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("sentiment_model")

st.title("Drug Review Sentiment Analyzer")

user_input = st.text_area("Enter review:")

if st.button("Predict"):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()
        confidence = torch.max(probs).item()

    sentiment = "Positive" if pred == 1 else "Negative"
    st.markdown(f"Sentiment: {sentiment}")
    st.markdown(f"Confidence: {confidence:.2f}")

# streamlit run final.py