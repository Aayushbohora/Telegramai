from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = Flask(__name__)

# Load a small GPT model with low memory usage
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Use low_cpu_mem_usage=True to fit in 512 MB
model = AutoModelForCausalLM.from_pretrained(
    "distilgpt2",
    low_cpu_mem_usage=True
)

@app.route("/")
def home():
    return "Tiny DistilGPT2 Chatbot is running!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")

    if not user_input.strip():
        return jsonify({"response": "Please send a message."})

    # Encode user input
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # Generate response with short max_length to save RAM
    chat_history_ids = model.generate(
        input_ids,
        max_length=100,       # keep it small for Render
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        top_k=50
    )

    # Decode response
    bot_response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render provides PORT env variable
    app.run(host="0.0.0.0", port=port)
