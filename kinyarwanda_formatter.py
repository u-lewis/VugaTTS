"""
Custom Coqui-TTS dataset formatter for the Kinyarwanda TTS dataset.

Coqui's `load_tts_samples` expects a formatter function with the signature:
    formatter(root_path, meta_file, **kwargs) -> List[Dict]

Each returned dict must contain at least:
    - "text": the transcript
    - "audio_file": absolute or root_path-relative path to the wav
    - "speaker_name": speaker id (single-speaker dataset, so a constant string)
    - "root_path": echoed back, required by Coqui internally

Our filelists (data/train_filelist_22050.txt, data/val_filelist_22050.txt)
are plain lines of:
    full/path/to/audio.wav|transcript text

This formatter reads that format directly - no LJSpeech-style
"filename|text|normalized_text" assumption, no implicit "wavs/" subfolder.
"""

import os
from typing import Dict, List


def kinyarwanda_formatter(root_path: str, meta_file: str, **kwargs) -> List[Dict]:
    """Load samples from a path|text filelist.

    Args:
        root_path: dataset root (BaseDatasetConfig.path). Not required for
            resolving audio paths here since our filelist already stores
            full paths, but Coqui expects it echoed back per-sample.
        meta_file: filename of the filelist (BaseDatasetConfig.meta_file_train
            or meta_file_val), resolved relative to root_path.
    """
    meta_path = os.path.join(root_path, meta_file)
    items = []

    with open(meta_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split("|", 1)
            if len(parts) != 2:
                print(f"[kinyarwanda_formatter] Skipping malformed line {line_num}: {line!r}")
                continue

            audio_file, text = parts
            text = text.strip()

            if not os.path.isfile(audio_file):
                print(f"[kinyarwanda_formatter] Missing audio file at line {line_num}: {audio_file}")
                continue

            items.append(
                {
                    "text": text,
                    "audio_file": audio_file,
                    "speaker_name": "kinyarwanda_speaker",
                    "root_path": root_path,
                }
            )

    return items
