from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch

model_path = "./toxic_bert_model/toxicbert_final_model/"

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    probs = torch.softmax(outputs.logits, dim=1)
    predicted_label = torch.argmax(probs, dim=1).item()
    return predicted_label, probs.tolist()


if __name__ == "__main__":
    text = "I hate the black japanese"
    # text = "I am gonna kill you"
    label, probs = predict(text)
    mapping = ['Toxic', 'severe toxic', 'obscence', 'threat', 'insult', 'identity hate']
    print(f"Probabilities: {probs}")
    print(f"Predicted Label: {mapping[label]}")


