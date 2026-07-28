from huggingface_hub import snapshot_download

print("Downloading Kinyarwanda TTS dataset...")

snapshot_download(
    repo_id="mbazaNLP/kinyarwanda-tts-dataset",
    repo_type="dataset",
    local_dir="data/raw/kinyarwanda-tts-dataset",
    local_dir_use_symlinks=False,
)

print("Download complete!")