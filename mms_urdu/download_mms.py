from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "facebook/mms-tts-urd-script_arabic"

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "mms_urdu"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading MMS Urdu model...")
print(f"Destination: {MODEL_DIR}")

downloaded_path = snapshot_download(
    repo_id=MODEL_ID,
    local_dir=MODEL_DIR,

    # The repository contains both model.safetensors and
    # pytorch_model.bin. We only need the safer SafeTensors file.
    ignore_patterns=[
        "pytorch_model.bin",
        "*.h5",
        "*.msgpack",
    ],
)

print("\nDownload completed successfully.")
print(f"Model saved at: {downloaded_path}")