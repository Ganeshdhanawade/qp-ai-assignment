from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model
def load_model():
    model_name = "EleutherAI/gpt-neo-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return {"model": model, "tokenizer": tokenizer}

# Query the model
def query_document(query: str, vector_store, model_data):
    tokenizer = model_data["tokenizer"]
    model = model_data["model"]
    
    # Retrieve relevant chunks from vector store
    relevant_chunks = vector_store.get_relevant_chunks(query)

    if not relevant_chunks:
        return None

    # Prepare input for the model
    input_text = " ".join(relevant_chunks) + "\nQuestion: " + query
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate the answer
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=150)
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer