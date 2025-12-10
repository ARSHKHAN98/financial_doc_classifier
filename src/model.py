from transformers import DistilBertForSequenceClassification

def create_model(num_labels: int):
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels
    )
    return model
