# Local Text-to-Speech Models

This project runs two text-to-speech models locally:

- **Kokoro-82M** for lightweight, natural English speech.
- **Meta MMS Urdu** for Urdu speech generation.

Model files are downloaded once and then loaded from local storage.

## Models

### Meta MMS Urdu

**Model:** `facebook/mms-tts-urd-script_arabic`

Use this first for Urdu TTS. It accepts Urdu in Arabic script and generates audio directly:

```text
Urdu text -> VITS -> WAV audio
```

Example: `السلام علیکم، میں ٹیکسٹ ٹو اسپیچ سیکھ رہا ہوں۔`

VITS is an end-to-end architecture, so it does not need a separate HiFi-GAN vocoder. The model has **36.3 million parameters**; its main weights are about **145 MB**. The complete repository is about **291 MB** because it includes the same weights in both SafeTensors and PyTorch formats, while Transformers normally loads only one.

Its main advantages are proper Urdu Arabic-script support and a relatively small model size.

### Kokoro-82M

**Model:** `hexgrad/Kokoro-82M`

Use Kokoro as the main English TTS model. It is small, fast, provides multiple male and female voices, and supports adjustable speech speed:

```text
English text -> Select voice -> Select speed -> WAV audio
```

Kokoro has **82 million parameters**. Its main weight file is about **327 MB**, and the complete repository with voices is about **363 MB**.

It provides voices for American and British English, Japanese, Mandarin Chinese, Spanish, French, Hindi, Italian, and Brazilian Portuguese. Sample voices included here are `af_heart`, `am_michael`, and `bf_emma`.


## Project Structure

```text
kokoro/
  download_kokoro.py   Download the model and sample voices
  test_kokoro.py       Generate English speech

mms_urdu/
  download_mms.py      Download the Urdu model
  test_mms.py          Generate Urdu speech
```

## Setup

Python 3.11 is recommended.

```powershell
python -m pip install -r requirements.txt
```

Kokoro may also require [eSpeak NG](https://github.com/espeak-ng/espeak-ng) for English phonemization.

## Usage

Download the models:

```powershell
python kokoro/download_kokoro.py
python mms_urdu/download_mms.py
```

Generate test audio:

```powershell
python kokoro/test_kokoro.py
python mms_urdu/test_mms.py
```

Generated WAV files are saved in each model's `outputs` directory.
