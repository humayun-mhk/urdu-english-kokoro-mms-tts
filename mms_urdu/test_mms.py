from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import write
from transformers import AutoTokenizer, VitsModel, set_seed


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "mms_urdu"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_DIR.exists():
    raise FileNotFoundError(
        "MMS model was not found. Run download_mms.py first."
    )

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")
print("Loading MMS Urdu model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

model = VitsModel.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
).to(device)

model.eval()

# Write numbers as Urdu words for the first test.
test_texts = [
    "السلام علیکم، میرا نام ہمایوں ہے۔",
    "میں مصنوعی ذہانت اور ٹیکسٹ ٹو اسپیچ ٹیکنالوجی سیکھ رہا ہوں۔",
    "آج میں تین مختلف آواز بنانے والے ماڈلز کا تجربہ کر رہا ہوں۔",
]

for index, text in enumerate(test_texts, start=1):
    print(f"\nGenerating audio {index}")
    print(f"Text: {text}")

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    inputs = {
        name: tensor.to(device)
        for name, tensor in inputs.items()
    }

    # VITS can generate slightly different rhythms each time.
    set_seed(1000 + index)

    with torch.inference_mode():
        output = model(**inputs)
        waveform = output.waveform[0]

    audio = waveform.detach().cpu().numpy().astype(np.float32)

    output_path = OUTPUT_DIR / f"mms_urdu_{index}.wav"

    write(
        filename=output_path,
        rate=model.config.sampling_rate,
        data=audio,
    )

    print(f"Saved: {output_path}")

print("\nAll Urdu audio files generated successfully.")