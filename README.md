# Math-OCR-project
In this repository, you will find my Math  OCR project and comments attached to it. As my first project, I acknowledge that the code may be low-quality, but I hope you have as much fun wandering through it as I did.

## Key notes about the project
This project is far from complete. In the future, I would like to train the CNN on all samples of the Hugging Face dataset linked below. The final loss for training on the first 135000 samples of the dataset is 0.189, and thus the model isn't completely accurate. Adding Beam Search is also something I have on my to-do list for this project.

# Neural Network-Based Math OCR

A custom deep learning pipeline built from scratch in PyTorch to convert handwritten mathematical equations into executable LaTeX strings. Developed and trained as an independent machine learning engineering project.

## 🚀 Key Features
- **Custom Architecture:** Built using a CNN Vision Encoder coupled with a Transformer Decoder sequence-to-sequence model.
- **Data & Scaling:** Trained progressively up to **135,000 samples** from the Hugging Face `deepcopy/MathWriting-human` dataset.
- **Robust Inference Pipeline:** Implements autoregressive greedy decoding, custom tokenization, and **aspect-ratio-preserving letterboxing (`ImageOps.pad`)** to handle arbitrary image dimensions without spatial distortion.
- **Mathematical Integration:** Automatically bridges raw OCR text into symbolic math via SymPy for evaluation or computational solving.

## 🛠️ Tech Stack
- **Deep Learning:** PyTorch, Torchvision
- **Datasets & NLP:** Hugging Face `datasets`
- **Environment:** Google Colab (GPU/TPU acceleration)
## 📂 Project Structure
```text
├── src/
│   ├── model.py         # CNN Encoder + Transformer Decoder definition
│   ├── dataset.py       # Data loading and preprocessing pipelines
│   └── tokenizer.py     # Custom character-level LaTeX tokenizer
├── math_ocr_model_135k.pth # Trained model weights (~2.5 MB)
├── inference.py         # Single-image inference script with letterboxing 


├── evaluate.py          # Evaluation script for Character Error Rate (CER)
└── requirements.txt     # Python dependencies
