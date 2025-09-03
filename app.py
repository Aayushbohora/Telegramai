from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = Flask(__name__)

# Load DistilGPT2 (smaller, faster)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained(
    "distilgpt2",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True  # fits in Render RAM
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

    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    chat_history_ids = model.generate(
        input_ids,
        max_length=150,  # smaller to save RAM
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        top_k=50
    )

    bot_response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
