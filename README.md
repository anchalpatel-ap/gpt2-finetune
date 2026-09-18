# gpt2-finetune

# Fine-Tuning GPT-2 for Context-Based Question Answering

## Overview
This minor project demonstrates how to fine-tune the base **GPT-2 (124M)** language model to perform Extractive Question Answering using the **SQuAD 2.0** (Stanford Question Answering Dataset). Instead of open-ended conversational generation, the model is trained to read a provided context paragraph and extract the exact factual answer to a given question.

## Features
* **Custom Data Formatting:** Converts standard CSV data into a strict `Context / Question / Answer` prompt structure.
* **Loss Masking:** Implements PyTorch label masking (`-100`) on padding tokens so the model learns facts instead of padding.
* **Hugging Face Trainer API:** Utilizes `TrainingArguments` and `Trainer` for optimized gradient accumulation and mixed-precision (FP16) training.
* **Aggressive Inference Trimming:** A custom post-processing pipeline that prevents base GPT-2 from experiencing "run-on generation" or hallucinating subsequent conversational turns.

## Prerequisites & Installation
This project is optimized to run on a **Google Colab T4 GPU**.

### Required Libraries
```bash
pip install torch pandas transformers datasets accelerate
