
Embedding model: https://huggingface.co/medicalai/ClinicalBERT


```
all_embeddings = []

for sentence in sentences:
    # Use device for tokenizer inputs
    inputs = tokenizer(
        sentence, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=512 # Ensure consistent max length
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']

    # Correctly compute mean-pooling while taking the attention mask into account
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) # Compute Weighted Sum of Token Embeddings
    
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9) # Compute Number of Real Tokens
    embedding = sum_embeddings / sum_mask

    # L2 Normalize the embeddings (Standard practice for cosine similarity)
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1) 
    
    all_embeddings.append(embedding.cpu().numpy())

```
