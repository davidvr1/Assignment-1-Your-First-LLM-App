from transformers import pipeline
import torch

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

generator = pipeline("text-generation", model=MODEL, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

prompt = "Give me a recipe for gluten-free pizza."

messages = [{"role": "user", "content": prompt}]
result = generator(messages, max_length=200, do_sample=True, temperature=0.7)

reply = result[0]["generated_text"][-1]["content"]
print(reply)

RESULTS_FILE = "qa_results.txt"
with open(RESULTS_FILE, "a", encoding="utf-8") as f:
    f.write(f"Model: {MODEL}\nPrompt: {prompt}\nAnswer: {reply}\n\n")
print(f"[saved to {RESULTS_FILE}]")
