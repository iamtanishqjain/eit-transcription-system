# 🎙️ EIT Audio Transcription System
### Audio-to-Text for Second/Additional Language Learner Data

---

## Project Overview

This project builds a robust, automated pipeline to convert Spanish **Elicited Imitation Task (EIT)** audio recordings from second-language learners into accurate text transcriptions — achieving **≥90% agreement with human transcribers**.

### The Problem
- Commercial speech-to-text tools fail on non-native speaker audio
- Learner speech has accents, disfluencies, transfer errors, and partial repetitions
- Manual transcription is slow and expensive for large research datasets

### Our Solution
A 3-stage pipeline:
```
Audio → [Preprocess] → [Transcribe] → [Post-process] → Text Output
```

---

## 📁 Project Structure

```
eit_transcription/
├── src/
│   └── transcription_pipeline.py    ← Core pipeline (all logic here)
├── notebooks/
│   └── EIT_Transcription_Notebook.ipynb  ← Interactive walkthrough
├── audio_samples/                   ← Put your .wav files here
├── outputs/                         ← Results saved here (CSV + JSON)
├── app.py                           ← Web interface
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### Option A: Run the Python Pipeline

```bash
# 1. Install dependencies
pip install openai-whisper numpy scipy flask

# 2. Put your .wav files in audio_samples/
# (or run with demo mode - no audio needed)

# 3. Run pipeline
python src/transcription_pipeline.py --audio_dir audio_samples --demo

# 4. Results saved to outputs/transcriptions.csv
```

### Option B: Web Interface

```bash
pip install flask openai-whisper
python app.py
# Open http://localhost:5000
```

### Option C: Jupyter Notebook

```bash
pip install jupyter openai-whisper pandas matplotlib
jupyter notebook notebooks/EIT_Transcription_Notebook.ipynb
```

---

## 🔧 Pipeline Stages

### Stage 1: Audio Preprocessing (`AudioPreprocessor`)
| Step | What it does |
|------|-------------|
| **Load** | Reads .wav file at 16kHz mono |
| **Normalize** | Scales amplitude to consistent peak (0.9) |
| **Noise Gate** | Zeros out background noise below threshold |
| **Trim Silence** | Removes leading/trailing silence using energy detection |

### Stage 2: Transcription (`WhisperTranscriber`)
- Uses **OpenAI Whisper** (multilingual ASR model)
- Language set to Spanish (`es`)
- Supports fine-tuned models via `model_path`
- **Demo mode** available (simulates realistic learner output without GPU)

| Model | Speed | Accuracy | Recommended For |
|-------|-------|----------|-----------------|
| tiny  | ⚡⚡⚡ | ★★☆ | Quick testing |
| base  | ⚡⚡  | ★★★ | Low-resource machines |
| small | ⚡   | ★★★★ | Balanced |
| **medium** | — | **★★★★★** | **Research use** |
| large | 🐌   | ★★★★★ | Maximum accuracy |

### Stage 3: Post-Processing (`LearnerLanguagePostProcessor`)
Corrects predictable errors in learner transcriptions:
- **Phonological confusions**: b/v, ll/y, c/s/z (common in Spanish learners)
- **Disfluency removal**: um, eh, uh, mmm
- **L1 transfer errors**: English words appearing in Spanish responses
- **Accent restoration**: Adds missing accent marks (está, allá, etc.)

---

## 📊 Output Format

### CSV (`outputs/transcriptions.csv`)
```
learner_id, sentence_id, raw_transcription, corrected_transcription,
confidence_score, duration_seconds, processing_time_ms, flags
```

### JSON (`outputs/transcriptions.json`)
```json
[
  {
    "learner_id": "learner_01",
    "sentence_id": "learner_01_sentence_01",
    "raw_transcription": "El niño come una manzana rojo",
    "corrected_transcription": "El niño come una manzana rojo",
    "confidence_score": 0.812,
    "duration_seconds": 2.34,
    "processing_time_ms": 310.5,
    "flags": []
  }
]
```

---

## 🎯 Meeting the 90% Agreement Target

The pipeline measures word-level F1 agreement between the system output and reference transcriptions.

**Strategies to reach ≥90%:**

1. **Fine-tune Whisper** on your actual learner dataset (see notebook Step 8)
2. **Expand post-processing rules** for your specific learner L1 backgrounds
3. **Use Whisper medium or large** (not tiny/base)
4. **Increase audio quality** during data collection (quiet room, close microphone)
5. **Add language model re-scoring** as a 4th pipeline stage

---

## 🔬 Fine-Tuning for Real Data

When you collect real EIT audio:

```python
# Prepare your dataset as a CSV:
# | audio_path | reference_text | learner_id | L1_background |

# Then fine-tune (see notebook for full code):
from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium')
# ... training code in notebook Step 8

# Use fine-tuned model:
pipeline = EITPipeline(use_demo=False, model_path='./whisper-eit-finetuned')
```

---

## 📋 Requirements

```
openai-whisper>=20231117
numpy>=1.24
scipy>=1.10
flask>=3.0
torch>=2.0       # for real Whisper (CPU or GPU)
transformers>=4.35  # for fine-tuning
pandas>=2.0      # optional, for notebook analysis
matplotlib>=3.7  # optional, for notebook plots
```

Install: `pip install -r requirements.txt`

---

## 👥 Naming Convention for Audio Files

For best results, name files:
```
{learner_id}_sentence_{N}.wav
```
Example: `learner_01_sentence_03.wav`

This allows the pipeline to automatically match files to EIT sentence indices.

---

## 📈 Expected Results

| Metric | Demo Mode | With Whisper Medium | With Fine-Tuned Model |
|--------|-----------|--------------------|-----------------------|
| Word Agreement | ~85% (simulated) | ~80-88% | **≥90% (target)** |
| Processing Speed | ~50ms/file | ~2-5s/file (CPU) | ~2-5s/file |
| Disfluency Detection | ✅ | ✅ | ✅ |
| L1 Transfer Correction | ✅ | ✅ | ✅ |

---

*Project for: Audio-to-Text Transcription for Second/Additional Language Learner Data*  
*Duration: 175 hours | Target: 90% agreement with human transcribers*
