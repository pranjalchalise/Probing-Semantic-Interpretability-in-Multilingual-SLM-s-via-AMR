from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states = True).to(device)
model.eval()


def get_cls_embeddings(sentences, batch_size=16):
    all_embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]

        encoder = tokenizer(batch,
                            padding = True,
                            truncation = True,
                            return_tensors = "pt").to(device)
        
        with torch.no_grad():
            outputs = model(**encoder)
            hidden_states = outputs.hidden_states #(13 layers of distilBERT)

            layer_cls = [h[:, 0, :].cpu().numpy() for h in hidden_states] #(b_s, 768)

        batch_embeddings = np.stack(layer_cls, axis = 1)
        all_embeddings.append(batch_embeddings)


    return np.concatenate(all_embeddings, axis = 0) #(shape: snts, 13, 768)