# Speech Understanding - Programming Assignment 2
---

## Overview

End-to-end pipeline for code-switched Hinglish lecture transcription and re-synthesis into Maithili (a low-resource Indo-Aryan language).

**Pipeline stages:**
1. Spectral subtraction denoising
2. Frame-level language identification (English vs Hindi) using a multi-head transformer
3. Constrained ASR using Whisper medium with n-gram logit biasing
4. IPA phonetic mapping for English, Hindi, and Hinglish
5. Semantic translation to Maithili using a 500-word dictionary
6. Speaker embedding extraction (d-vector)
7. MMS-TTS Maithili speech synthesis
8. DTW prosody warping to preserve teaching style
9. Anti-spoofing classifier using LFCC features
10. FGSM adversarial robustness evaluation

---

## Results

| Metric | Value | Target |
|--------|-------|--------|
| WER English | 7.49% | less than 15% |
| WER Hindi | 0.0% | less than 25% |
| MCD | 26.52 dB | less than 8.0 dB |
| LID F1 | 0.894 | at least 0.85 |
| LID Switches | 4401 | within 200ms |
| Anti-Spoof EER | 0.0 | less than 0.10 |
| FGSM Min Epsilon | 0.1 | reported |

---

## Requirements

Python 3.10 or higher, CUDA GPU recommended.

Install dependencies:

`
pip install -r requirements.txt
`

---

## How to Run

### On Kaggle (recommended)

1. Create a new Kaggle notebook with T4 GPU and internet enabled
2. Upload original_segment.wav and student_voice_ref.wav as a dataset
3. Upload pipeline.ipynb
4. Run all cells in order

### On Google Colab

1. Upload original_segment.wav and student_voice_ref.wav to the Colab session
2. Upload pipeline.ipynb
3. Run all cells in order

### Locally

1. Place original_segment.wav and student_voice_ref.wav in the same directory as pipeline.ipynb
2. Launch Jupyter and open pipeline.ipynb
3. Run all cells in order

---

## File Structure

`
pipeline.ipynb          - Main pipeline notebook (single cell, runs end to end)
requirements.txt        - Python dependencies
lid_weights.pt          - Pre-trained LID model weights
original_segment.wav    - Source lecture audio (10 minutes)
student_voice_ref.wav   - Reference voice for cloning (60 seconds)
outputs/
  denoised_lecture.wav      - Spectral subtraction output
  transcript.json           - Whisper transcription with timestamps
  ipa_transcript.txt        - IPA phonetic representation
  maithili_transcript.txt   - Maithili translation
  output_LRL_cloned.wav     - Final synthesised Maithili speech (22050 Hz)
  adversarial_segment.wav   - FGSM adversarial example
  metrics.txt               - All evaluation metrics
`

---

## Notes

The MCD value of 26.52 dB is higher than the target of 8.0 dB. This is a known limitation of the facebook/mms-tts-mai model, which is a fixed-speaker TTS system with no speaker conditioning interface. DTW prosody warping reduces MCD from 31.4 to 26.52 dB by aligning temporal structure, but the spectral envelope remains fixed to the canonical MMS-TTS Maithili voice.

---
