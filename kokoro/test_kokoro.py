from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from kokoro import KModel, KPipeline


MODEL_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24_000

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "kokoro_82m"
OUTPUT_DIR = BASE_DIR / "outputs"

CONFIG_PATH = MODEL_DIR / "config.json"
WEIGHTS_PATH = MODEL_DIR / "kokoro-v1_0.pth"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

required_files = [
    CONFIG_PATH,
    WEIGHTS_PATH,
    MODEL_DIR / "voices" / "af_heart.pt",
    MODEL_DIR / "voices" / "am_michael.pt",
    MODEL_DIR / "voices" / "bf_emma.pt",
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    raise FileNotFoundError(
        "Some Kokoro files are missing. "
        "Run download_kokoro.py first.\n"
        + "\n".join(missing_files)
    )

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")
print("Loading Kokoro model...")

# Load the model from the local downloaded files.
model = KModel(
    repo_id=MODEL_ID,
    config=str(CONFIG_PATH),
    model=str(WEIGHTS_PATH),
).to(device).eval()

# American and British pipelines use the same speech model.
american_pipeline = KPipeline(
    lang_code="a",
    repo_id=MODEL_ID,
    model=model,
)

british_pipeline = KPipeline(
    lang_code="b",
    repo_id=MODEL_ID,
    model=model,
)


def generate_audio(
    pipeline: KPipeline,
    text: str,
    voice_path: Path,
    speed: float,
    output_name: str,
) -> None:
    """Generate all Kokoro chunks and combine them into one WAV file."""

    voice_tensor = torch.load(
        voice_path,
        map_location="cpu",
        weights_only=True,
    )

    generated_parts: list[np.ndarray] = []

    generator = pipeline(
        text,
        voice=voice_tensor,
        speed=speed,
    )

    for chunk_number, (_, phonemes, audio) in enumerate(
        generator,
        start=1,
    ):
        print(f"Generated chunk {chunk_number}")
        print(f"Phonemes: {phonemes}")

        if torch.is_tensor(audio):
            audio = audio.detach().cpu().numpy()
        else:
            audio = np.asarray(audio)

        generated_parts.append(
            audio.astype(np.float32)
        )

    if not generated_parts:
        raise RuntimeError("Kokoro did not generate any audio.")

    complete_audio = np.concatenate(generated_parts)

    output_path = OUTPUT_DIR / output_name

    sf.write(
        output_path,
        complete_audio,
        SAMPLE_RATE,
    )

    print(f"Saved: {output_path}\n")


test_text = (
    "Hello, this is a local text to speech experiment. "
    "I am testing different speakers, pronunciation, rhythm, "
    "and speaking speed using the Kokoro speech model."
)

# American female voice at normal speed.
generate_audio(
    pipeline=american_pipeline,
    text=test_text,
    voice_path=MODEL_DIR / "voices" / "af_heart.pt",
    speed=1.0,
    output_name="kokoro_american_female.wav",
)

# American male voice at faster speed.
generate_audio(
    pipeline=american_pipeline,
    text=test_text,
    voice_path=MODEL_DIR / "voices" / "am_michael.pt",
    speed=1.25,
    output_name="kokoro_american_male_fast.wav",
)

# British female voice at slower speed.
generate_audio(
    pipeline=british_pipeline,
    text=test_text,
    voice_path=MODEL_DIR / "voices" / "bf_emma.pt",
    speed=0.90,
    output_name="kokoro_british_female_slow.wav",
)

print("All Kokoro tests completed successfully.")