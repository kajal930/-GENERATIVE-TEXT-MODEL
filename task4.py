# --- GPT-2 TEXT GENERATION --

from transformers import pipeline, set_seed
print("\n===== GPT-2 TEXT GENERATION =====\n")

generator = pipeline('text-generation', model='gpt2')
set_seed(42)

prompt = "The future of artificial intelligence is"
gpt_output = generator(prompt, max_length=80, num_return_sequences=1)

print("🔮 GPT-2 Output:\n", gpt_output[0]['generated_text'])

# --- LSTM TEXT GENERATION FROM SCRATCH ---

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("\n===== LSTM TEXT GENERATION =====\n")

# Small sample data
data = """Machine learning is a field of artificial intelligence. It gives computers the ability to learn without being explicitly programmed. Deep learning is a subset of machine learning involving neural networks with many layers."""

# Preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in data.split('.'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(2, len(token_list)):
        input_sequences.append(token_list[:i+1])

max_seq_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre'))

X = input_sequences[:, :-1]
y = tf.keras.utils.to_categorical(input_sequences[:, -1], num_classes=total_words)

# Build Model
model = Sequential()
model.add(Embedding(total_words, 10, input_length=max_seq_len-1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=0)

# Generate Text
def generate_text(seed_text, next_words=10):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                seed_text += " " + word
                break
    return seed_text

# Example Output
print("\n🧠 LSTM Generated Text:\n", generate_text("Machine learning", next_words=10))




