from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "hexgrad/Kokoro-82M"

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "kokoro_82m"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading Kokoro model and selected voices...")
print(f"Destination: {MODEL_DIR}")

downloaded_path = snapshot_download(
    repo_id=MODEL_ID,
    local_dir=MODEL_DIR,

    # Download only the files needed for our tests.
    allow_patterns=[
        "config.json",
        "kokoro-v1_0.pth",
        "voices/af_heart.pt",
        "voices/am_michael.pt",
        "voices/bf_emma.pt",
    ],
)

print("\nDownload completed successfully.")
print(f"Model saved at: {downloaded_path}")

print("\nDownloaded voices:")
print("1. af_heart   - American female")
print("2. am_michael - American male")
print("3. bf_emma    - British female")