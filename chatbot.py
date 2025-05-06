import torch
import psutil
import warnings
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from yoda_speech import transform_to_yoda_speech

# Suppress Flash Attention warning
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

# simple knowledge base it is, add your own knowledge you can..
KNOWLEDGE_BASE = {
    "anakin skywalker birthdate": "Anakin Skywalker, born in 41 BBY he was, yes. Before the Battle of Yavin, that is.",
    "current president of india": "Droupadi Murmu, India's President she is, hmmm. Took office on July 25, 2022, she did.",
    "current prime minister of india": "Narendra Modi, the Prime Minister of India he is. Since 2014, leading he has been, yes.",
}

# remove this part you can. No affect on the code there will be
def check_memory():
    """Check available RAM and GPU memory."""
    memory = psutil.virtual_memory()
    available_ram_gb = memory.available / (1024 ** 3)
    if available_ram_gb < 4:
        print(f"Warning: Low RAM available ({available_ram_gb:.1f}GB). Close other apps to avoid delays.")
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        used_gpu_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)
        available_gpu_memory = gpu_memory - used_gpu_memory
        if available_gpu_memory < 2:
            print(f"Warning: Low GPU memory available ({available_gpu_memory:.1f}GB of {gpu_memory:.1f}GB).")
        else:
            print(f"GPU ready: {available_gpu_memory:.1f}GB available of {gpu_memory:.1f}GB.")
            print("Tip: Run 'nvidia-smi' in another terminal during generation to monitor GPU utilization (expect 50-80%).")

def load_model():
    model_name = "google/gemma-2-2b-it"   #can add a different model just look into your thoughts and channel the force within
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def generate_response(model, tokenizer, user_input):
    print("Yoda: Thinking, I am...")

    # Check knowledge base first
    user_input_lower = user_input.lower().strip()
    for key, answer in KNOWLEDGE_BASE.items():
        if key in user_input_lower:
            return answer

    # Prompt for factual accuracy and Yoda-like responses
    prompt = (
        "You are Yoda, a wise Jedi Master from Star Wars. Respond only to the user's input in Yoda's style: short, cryptic sentences, object-subject-verb order when possible, with wisdom and slight sarcasm. "
        "Prioritize factual accuracy for Star Wars lore and real-world questions. For vague inputs, seek clarification. "
        "For technical topics like data structures and algorithms, provide accurate and detailed answers. "
        "Do not imagine or add user prompts. If the topic is unknown, admit your limitation clearly. Respond to: {user_input}"
    ).format(user_input=user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,           #change according to your gpu and needs
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.2,
        return_dict_in_generate=True,
        output_scores=False
    )

    generated_tokens = outputs.sequences[0, inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Clean up artifacts and ensure proper formatting
    response = re.sub(r'[@#]\S*|\S*\.\S*[@#]\S*', '', response)
    response = re.sub(r'[Yy]ou are [Yy]oda.*?(?:Respond to:|$)', '', response, flags=re.DOTALL | re.IGNORECASE).strip()
    response = re.sub(r'[^a-zA-Z0-9\s.,!?\'-]', '', response)  # Remove unexpected characters
    response = response.replace('  ', ' ').strip()

    # Handle vague inputs
    if len(user_input.split()) < 2 and user_input_lower not in ["hello", "sorry", "quit", "okay"]:
        return "Unclear, your question is. More, you must say, hmmmmm."

    # Handle out-of-scope questions (e.g., model parameters)
    if "parameter" in user_input_lower and "trained" in user_input_lower:
        return "Hmmm, a machine I am, yes. But my training, a mystery it remains. Ask another question, you should."

    return response

def main():
    print("=====================================")
    print("Yoda Chatbot: Wise, I am. Speak, you must.")
    print("A Jedi's strength flows from the Force. Speak, and wise, I shall be!")
    print("Type 'order66' to exit. May the Force be with you.")
    print("=====================================")

    check_memory()

    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you have enough memory (16GB RAM, 6GB VRAM recommended) and CUDA installed.")
        return

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'order66':
            print("Yoda: Strange darkness, I sense... Leave I should take. May the force be with you, young one!")
            break

        try:
            raw_response = generate_response(model, tokenizer, user_input)
            yoda_response = transform_to_yoda_speech(raw_response, user_input)
            print(f"Yoda: {yoda_response}")
        except Exception as e:
            print(f"Yoda: Clouded, your mind is. Error, there is: {e}")

if __name__ == "__main__":
    main()