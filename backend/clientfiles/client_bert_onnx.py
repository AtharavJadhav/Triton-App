import numpy as np
import tritonclient.http as httpclient
from transformers import BertTokenizer
import os
import random
import time
from nltk.translate.bleu_score import sentence_bleu

# Define the server URL and model name
url = "localhost:8000"
model_name = "bert_model_onnx"

# Create a Triton HTTP client
client = httpclient.InferenceServerClient(url=url)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')

# Prepare a sample input text
def prepare_text(input_text):
    encoded_input = tokenizer(input_text, return_tensors='np', padding='max_length', max_length=128, truncation=True)
    input_ids = encoded_input['input_ids']
    return input_ids

# Function to load all input texts and expected outputs
def load_texts(base_path):
    texts = []
    with open(base_path, 'r') as file:
        for line in file:
            text = line.strip()
            texts.append(text)
    return texts

# Shuffle and create a list of texts
base_path = "/home/atharav/react/model/data/test_data/bert_model/text_generation_data.txt"
texts = load_texts(base_path)
random.shuffle(texts)

# Measure accuracy, latency, and throughput
total_texts = len(texts)
start_time = time.time()
bleu_scores = []

for i in range(total_texts):
    input_text = texts[i]
    input_ids = prepare_text(input_text)

    # Create the input object
    input_ids_obj = httpclient.InferInput("input.1", input_ids.shape, "INT64")
    input_ids_obj.set_data_from_numpy(input_ids)

    # Create the output object
    outputs = httpclient.InferRequestedOutput("367")

    # Send the request to the server
    response = client.infer(model_name, inputs=[input_ids_obj], outputs=[outputs])

    # Get the results
    result = response.as_numpy("367")
    generated_text = tokenizer.decode(result.flatten(), skip_special_tokens=True)

    # Evaluate the generated text using BLEU score
    reference = [input_text.split()]
    candidate = generated_text.split()
    bleu_score = sentence_bleu(reference, candidate)
    bleu_scores.append(bleu_score)

end_time = time.time()

# Calculate BLEU score average, latency, and throughput
average_bleu = sum(bleu_scores) / total_texts
latency = (end_time - start_time) / total_texts
throughput = total_texts / (end_time - start_time)

# Check if results.txt exists and delete it before writing new results
results_file_path = "results.txt"
if os.path.exists(results_file_path):
    os.remove(results_file_path)

with open(results_file_path, "w") as f:
    f.write(f"Average BLEU Score: {average_bleu:.8f}\n")
    f.write(f"Latency: {latency:.8f} seconds per text\n")
    f.write(f"Throughput: {throughput:.3f} texts per second\n")

print(f"Average BLEU Score: {average_bleu:.8f}")
print(f"Latency: {latency:.8f} seconds per text")
print(f"Throughput: {throughput:.3f} texts per second")
