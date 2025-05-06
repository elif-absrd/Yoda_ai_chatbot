# Yoda Chatbot: Wisdom of the Force

It was May the 4th and I was bored, so why not create a simple yoda ai chat bot assistant.
A chatbot mimicking Yoda, the Jedi Master. Speak with you in Yoda's style, it will with short, cryptic sentences with object-subject-verb order, wisdom, and slight sarcasm. Powered by the Force (and AI),May the Force be with you as you embark on this journey!

## Project Overview

Built with Python and the `transformers` library, this chatbot uses the `google/gemma-2-2b-it` model (2B parameters) to generate responses. Optimized for GPU usage with PyTorch nightly and Flash Attention, it fits in 6GB VRAM and works smothly . A knowledge base ensures factual accuracy for common *Star Wars* and real-world queries, while `yoda_speech.py` transforms responses into Yoda’s iconic style.You can change the model according to your vram.

## Prerequisites

Before you begin, ensure your system is ready, young Padawan:
- **Python**: Version 3.8 or higher.
- **GPU**: NVIDIA GPU with at least 6GB VRAM and CUDA 12.1 or higher.
- **Operating System**: Windows (tested on Windows with PowerShell).
- **Hugging Face Account**: A token to access models (stored at `C:\Users\<your-username>\.cache\huggingface\token`).

## Setup Instructions

Follow these steps to awaken the Force within this chatbot:

1. **Clone the Repository** (if using Git):
   ```powershell
   git clone https://github.com/elif-absrd/Yoda_ai_chatbot
   cd chatbot_yoda
   ```

2. **Set Up a Virtual Environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**:
   - Save the provided `requirements.txt` to your project directory.
   - Install the dependencies:
     ```powershell
     pip install -r requirements.txt
     ```
   - Install PyTorch nightly for Flash Attention support (optimizes VRAM usage):
     ```powershell
     pip uninstall torch
     pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
     ```

4. **Authenticate with Hugging Face**:
   - Log in to Hugging Face to access the `google/gemma-2-2b-it` model:
     ```powershell
     huggingface-cli login
     ```
   - Enter your token when prompted (paste using Right-Click).

5. **Verify GPU Support**:
   - Ensure PyTorch is using your GPU:
     ```python
     import torch
     print(torch.cuda.is_available())
     print(torch.cuda.get_device_name(0))
     ```
   - Expected output:
     ```
     True
     NVIDIA GeForce RTX 4050 Laptop GPU
     ```

## Usage

Run the chatbot and speak with Yoda, you shall:

1. **Start the Chatbot**:
   ```powershell
   python .\chatbot.py
   ```

2. **Interact with Yoda**:
   - Type your question or greeting (e.g., “hello”, “what is a tree structure in DSA”, “who is the current president of India”).
   - To exit, type `quit`.

3. **Example Interactions**:
   ```
   You: hello
   Yoda: Mmm, greet me you do. Polite, you are, young one.
   ```
   ```
   You: what is birthdate of anakin skywalker in star wars
   Yoda: Anakin Skywalker, born in 41 BBY he was, yes. Before the Battle of Yavin, that is.
   ```
  

## Files in the Project

- **`chatbot.py`**: The main script to run the Yoda chatbot. Loads the model, generates responses, and applies Yoda’s speech style.
- **`yoda_speech.py`**: Transforms responses into Yoda’s iconic style with Easter eggs for *Star Wars* terms.
- **`requirements.txt`**: Lists dependencies, including PyTorch nightly installation instructions.

## Troubleshooting

If the dark side clouds your path, try these solutions:

- **CPU Offloading Warning (`Some parameters are on the meta device...`)**:
  - Check VRAM usage with `nvidia-smi`. If near 6GB, clear GPU memory:
    ```powershell
    nvidia-smi --gpu-reset
    ```
  - Reduce `max_new_tokens` in `chatbot.py` to 30:
    ```python
    max_new_tokens=30
    ```

- **Model Fails to Load**:
  - Ensure you’re logged in to Hugging Face:
    ```powershell
    huggingface-cli login
    ```
  - Clear the Hugging Face cache and retry:
    ```powershell
    Remove-Item -Recurse -Force $env:USERPROFILE\.cache\huggingface
    ```

- **Responses Are Vague or Incorrect**:
  - Add more entries to the `KNOWLEDGE_BASE` in `chatbot.py` for common questions.
  - Adjust the prompt to prioritize factual accuracy:
    ```python
    "Provide concise, factual answers in Yoda's style."
    ```

## Contributing

Wish to strengthen the Force within this project? Add more *Star Wars* lore to the `KNOWLEDGE_BASE`, improve Yoda’s speech patterns in `yoda_speech.py`, or optimize the model for better performance. Submit a pull request, you may!

## Problems
Even after prompting in the google model to speak like yoda, I am using yoda_speech.py to genrate a better response which might slows the processing time but with yoda_speech.py we can really improve the responses. 
One problem that persists is puntuation errors cause I am kindof rearranging the words so sometime you can find commas after a fullstop and some similar senario.

## Acknowledgments

- **Hugging Face**: For the `google/gemma-2-2b-it` model.
- **PyTorch Team**: For Flash Attention support in nightly builds.
- **Star Wars**: For the wisdom of Master Yoda, inspiring this project it did.

May the Force be with you, always!
