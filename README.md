🧠 **Machine Translation using Transformer (English → French & English → Spanish)**
===================================================================================

A Natural Language Processing (NLP) project that implements **sequence-to-sequence Transformer models** to translate English sentences into **Spanish** and (future-ready) **French**.This project includes:

✅ Transformer encoder–decoder architecture✅ Custom positional embedding, vectorizers, and tokenizers✅ English→Spanish working model❗ English→French model placeholder (requires retraining)🎨 Tkinter GUI for real-time translation📁 All saved models + vectorization layers

📂 **Project Structure**
------------------------
```
MACHINE_TRANSLATION/
│
├── english_to_french_model/
│   ├── assets/
│   ├── variables/
│   ├── saved_model.pb
│   └── keras_metadata.pb
│
├── eng-french.ipynb
├── eng-spanish.ipynb
│
├── english_to_french_model.keras
├── transformer_model.keras        # Spanish transformer model (main working model)
│
├── eng_vectorization_config.json
├── eng_vocab.json
├── spa_vectorization_config.json
├── spa_vocab.json
├── sequence_length.json
│
├── gui.py                         # Tkinter-based GUI translator
└── README.md
```



🚀 **Features**
---------------

### ✅ **English → Spanish Translation (Working)**

*   Trained on the **spa-eng dataset** from TensorFlow.
    
*   Uses a custom Transformer built from scratch (Keras + MultiHeadAttention).
    
*   Properly saved **vectorizers** + **vocab JSONs** for inference in GUI.
    

### ⚠️ **English → French Translation (Yet to be completed)**

*   Model structure ready.
    
*   Training required before integrating into GUI.
    

### 🖥️ **Interactive GUI using Tkinter**

*   Simple interface to type English text.
    
*   Choose language (currently: Spanish).
    
*   View translation on screen.
    

🧰 **Technologies Used**
------------------------

ComponentTechnologyLanguagePythonLibrariesTensorFlow, Keras, NumPy, TkinterModel TypeTransformer Encoder–DecoderVectorizationTextVectorization + custom standardizationUITkinter

📦 **Dataset**
--------------

This project uses the free **spa–eng parallel corpus**:

`   http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip   `

✔ Downloaded manually using keras.utils.get\_file()✔ Extracted to a fixed directory✔ Preprocessed:

*   Lowercase
    
*   Remove punctuation
    
*   Add \[start\] ... \[end\] tokens
    

🔧 **How the Model Works**
--------------------------

### 1️⃣ Vectorization

Custom TextVectorization layers are created for:

*   English
    
*   Spanish
    

✔ Saved as JSON✔ Loaded inside gui.py during inference

### 2️⃣ Encoder

*   Multi-head self-attention
    
*   Feed-forward dense blocks
    
*   Positional embeddings
    
*   Layer normalization + masking
    

### 3️⃣ Decoder

*   Causal attention mask
    
*   Self & cross attention
    
*   Dense projection
    
*   Softmax output
    

### 4️⃣ Training

Training pipeline (eng-spanish.ipynb) includes:

*   Preprocessing
    
*   Creating datasets using tf.data.Dataset.from\_tensor\_slices()
    
*   Model compile + fit
    
*   Saving transformer model + vectorizers
    

🖥️ Running the Project
-----------------------

### **1\. Clone the repository**

`   git clone https://github.com/Akash-8004/Machine-Translation.git  cd Machine-Translation   `

### **2\. Install dependencies**

`   pip install tensorflow keras numpy tk   `

### **3\. Run the GUI**

`   python gui.py   `

### GUI Preview

*   Input: English text
    
*   Output: Spanish translation via trained transformer
    

📌 **Future Improvements**
--------------------------

*   Train English→French model
    
*   Add beam search decoding
    
*   Add support for more languages
    
*   Improve GUI design
    
*   Package as a standalone application
