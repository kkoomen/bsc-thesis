from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
import torch
from threading import Thread
from dotenv import load_dotenv

load_dotenv()

model_id = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)

# Chat memory
chat_history = []

# Format system prompt
system_prompt = "<<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n"

def build_prompt(new_prompt):
    """Build full prompt with chat history"""
    conv = ""
    for prompt, response in chat_history:
        conv += f"[INST] {prompt} [/INST] {response} "
    return system_prompt + conv + f"[INST] {new_prompt} [/INST]"

print("💬 Chat with LLaMA 2 (type 'exit' to quit)")

while True:
    user_input = input("> ")
    if user_input.lower() in {"exit", "quit"}:
        break

    full_prompt = build_prompt(user_input)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # Streamer setup
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("Assistant:", end=" ", flush=True)
    response = ""
    for token in streamer:
        print(token, end="", flush=True)
        response += token

    chat_history.append((user_input, response))
    print("\n")
