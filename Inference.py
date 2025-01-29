from transformers import BertTokenizer, BertForSequenceClassification
import torch

from google.colab import drive

# Mount Google Drive to access it
drive.mount('/content/drive')

# Path to the local saved model and tokenizer in Google Drive
model_path = './model'
tokenizer_path = './tokenizer'

# Load the tokenizer and model from the saved paths
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Sample Amharic text
sample_text = "የገና በአል አከባበር ልዩ ዝግጅት?"

# Tokenize the input text
inputs = tokenizer(sample_text, return_tensors='pt', padding=True, truncation=True, max_length=256)

# Get the prediction
with torch.no_grad():
    outputs = model(**inputs)

# Extract the logits and predict the sentiment
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=1).item()

# Map the predicted class to sentiment (adjusted as per your mapping)
sentiment_dict = {0: 'neutral', 1: 'positive', 2: 'negative', 3: 'strongly positive', 4: 'strongly negative'}
predicted_sentiment = sentiment_dict[predicted_class]

print(f"Text: {sample_text} => Predicted Sentiment: {predicted_sentiment}")
