# import libraries
import tkinter as tk
from tkinter import ttk

from keras.layers import TextVectorization
import re
import tensorflow.strings as tf_strings
import json
import string
from keras.models import load_model
from tensorflow import argmax
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
import numpy as np
import tensorflow as tf

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from keras.saving import register_keras_serializable


# English to Spanish translation
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf_strings.lower(input_string)
    return tf_strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")


# Load the English vectorization layer configuration
with open('eng_vectorization_config.json') as json_file:
    eng_vectorization_config = json.load(json_file)


# Recreate the English vectorization layer with basic configuration
eng_vectorization = TextVectorization(
    max_tokens = eng_vectorization_config['max_tokens'],
    output_mode = eng_vectorization_config['output_mode'],
    output_sequence_length = eng_vectorization_config['output_sequence_length']
)

# Apply the custom standardization function
eng_vectorization.standardize = custom_standardization


# Load the Spanish vectorization layer configuration
with open('spa_vectorization_config.json') as json_file:
    spa_vectorization_config = json.load(json_file)


# Recreate the Spanish vectorization layer with basic configuration
spa_vectorization = TextVectorization(
    max_tokens = spa_vectorization_config['max_tokens'],
    output_mode = spa_vectorization_config['output_mode'],
    output_sequence_length = spa_vectorization_config['output_sequence_length'],
    standardize = custom_standardization
)


# Load and set the English vocabulary
with open('eng_vocab.json') as json_file:
    eng_vocab = json.load(json_file)
    eng_vectorization.set_vocabulary(eng_vocab)

# Load and set the Spanish vocabulary
with open('spa_vocab.json') as json_file:
    spa_vocab = json.load(json_file)
    spa_vectorization.set_vocabulary(spa_vocab)


@register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim = vocab_size, output_dim = embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim = sequence_length, output_dim = embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start = 0, limit = length, delta = 1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions
    
    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return tf.not_equal(inputs, 0)
        else:
            return None
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "sequence_length": self.sequence_length,
            "embed_dim": self.embed_dim,
        })
        return config


@register_keras_serializable()
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads = num_heads, key_dim = embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation = "relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True
        
    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, None, :], dtype = tf.int32)
        else:
            padding_mask = None
            
        attention_output = self.attention(
            query = inputs,
            value = inputs,
            key = inputs,
            attention_mask = padding_mask,
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
        })
        return config


@register_keras_serializable()
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads = num_heads, key_dim = embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads = num_heads, key_dim = embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(latent_dim, activation = "relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True
    def call(self, inputs, encoder_outputs, mask=None):
        casual_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, None, :], dtype = tf.int32)
            padding_mask = tf.minimum(padding_mask, casual_mask)
        else:
            padding_mask = None
            
        attention_output_1 = self.attention_1(
            query = inputs,
            value = inputs,
            key = inputs,
            attention_mask = casual_mask,
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)
        
        attention_output_2 = self.attention_2(
            query = out_1,
            value = encoder_outputs,
            key = encoder_outputs,
            attention_mask = padding_mask,
        )
        
        out_2 = self.layernorm_2(out_1 + attention_output_2)
        proj_output = self.dense_proj(out_2)
        
        return self.layernorm_3(out_2 + proj_output)
    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, None]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, tf.int32)
        mask = tf.reshape(mask,(1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [
                tf.expand_dims(batch_size, -1),
                tf.convert_to_tensor([1, 1]),
            ],
            axis = 0,
        )
        return tf.tile(mask, mult)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "latent_dim": self.latent_dim,
            "num_heads": self.num_heads,
        })
        return config


# load the spanish model
# transformer = load_model('transformer_model.keras')
transformer = load_model(
    'transformer_model.keras',
    custom_objects={
        'PositionalEmbedding': PositionalEmbedding,
        'TransformerEncoder': TransformerEncoder,
        'TransformerDecoder': TransformerDecoder
    }
)


spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20

def decode_sentence(input_sentence):
    tokenized_input_sentence = eng_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = spa_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = tf.argmax(predictions[0, i, :]).numpy().item(0)
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence





# English to French translation

# French model loading disabled due to compatibility issues
# TODO: Retrain and save the French model properly
model = None

# Tokenizers disabled until French model is fixed
english_tokenizer = None
french_tokenizer = None
max_length = None
    
def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')

def translate_to_french(english_sentence):
    return "French translation currently unavailable. Please retrain the model."

def translate_to_spanish(english_sentence):
    spanish_sentence = decode_sentence(english_sentence)
    print("Spanish translation: ", spanish_sentence)
    
    return spanish_sentence.replace("[start]", "").replace("[end]", "")

# Function to handle translation request based on selected language
def handle_translate():
    selected_language = language_var.get()
    english_sentence = text_input.get("1.0", "end-1c")
    
    if selected_language == "French":
        translation = translate_to_french(english_sentence)
    elif selected_language == "Spanish":
        translation = translate_to_spanish(english_sentence)
        
    translation_output.delete("1.0", "end")
    translation_output.insert("end", f"{selected_language} translation: {translation}")


# Setting up the main window
root = tk.Tk()
root.title("Language Translator")
root.geometry("550x600")

# Font configuration
font_style = "Times New Roman"
font_size = 14

# Frame for input
input_frame = tk.Frame(root)
input_frame.pack(pady=10)

# Heading for input
input_heading = tk.Label(input_frame, text="Enter the text to be translated", font=(font_style, font_size, 'bold'))
input_heading.pack()
# Text input for English sentence
text_input = tk.Text(input_frame, height=5, width=50, font=(font_style, font_size))
text_input.pack()

# Language selection
language_var = tk.StringVar()
language_label = tk.Label(root, text="Select the language to translate to", font=(font_style, font_size, 'bold'))
language_label.pack()
language_select = ttk.Combobox(root, textvariable=language_var, values=["Spanish"], font=(font_style, font_size), state="readonly")
language_select.pack()

# Submit button
submit_button = ttk.Button(root, text="Translate", command=handle_translate)
submit_button.pack(pady=10)

# Frame for output
output_frame = tk.Frame(root)
output_frame.pack(pady=10)
# Heading for output
output_heading = tk.Label(output_frame, text="Translation: ", font=(font_style, font_size, 'bold'))
output_heading.pack()

# Text output for translations
translation_output = tk.Text(output_frame, height=10, width=50, font=(font_style, font_size))
translation_output.pack()

# Running the application
root.mainloop()