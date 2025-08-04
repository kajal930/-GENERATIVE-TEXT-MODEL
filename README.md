# -GENERATIVE-TEXT-MODEL# ✨ CODETECH - TASK-4

**Name:** Kajal Kumari  
**Company:** CODETECH IT SOLUTIONS  
**Task ID:** CT04DH2042
**Domain:** Artificial Intelligence  
**Duration:** 5th July to 5Tth August 2025  
**Mentor:** Neela Santhosh Kumar  

---

## 🧠 Project Title: **Generative Text Model**

---

### 🎯 Objective  
To build a generative text model using deep learning techniques that can generate coherent and meaningful paragraphs or sentences based on user-provided prompts. The system is designed to mimic human-like writing and assist in creative or automated content generation.

---

### 🔑 Key Activities

- Prepare or load a dataset (optional for training from scratch).
- Use a **pre-trained language model** (e.g., GPT-2, GPT-Neo, LSTM).
- Accept text prompts as input from the user.
- Generate text output that continues or completes the given prompt.
- Develop a Python script that:
  - Accepts user input (prompt)
  - Returns AI-generated text output
- *(Optional)* Add a simple UI using Streamlit or Tkinter for interactivity.

---

### 🛠️ Technologies / Tools Used

- **Language:** Python  
- **Libraries / Frameworks:**
  - `transformers` – for loading pre-trained models like GPT-2
  - `torch` – backend for model execution
  - `streamlit` – (optional) for building a user interface
  - `nltk` – (optional) for any preprocessing or prompt filtering

---

### 🧾 Input
- A text prompt (e.g., "Once upon a time", "The future of AI is...")

### 📤 Output
- AI-generated paragraph or continuation based on the input

---

### ✅ Example Code Snippet
```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
prompt = "The future of artificial intelligence"
result = generator(prompt, max_length=100, num_return_sequences=1)

print("Generated Text:\n", result[0]['generated_text'])
