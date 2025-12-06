from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
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

            cls_embeds = outputs.last_hidden_state[:,0,:]

        all_embeddings.append(cls_embeds.cpu().numpy())


    return np.vstack(all_embeddings)