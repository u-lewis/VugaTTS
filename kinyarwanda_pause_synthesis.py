"""
Punctuation-aware synthesis for Kinyarwanda TTS.

Your model's character vocabulary intentionally excludes punctuation (only
letters, space, and apostrophe), so normalize_kinyarwanda_text() strips
punctuation entirely - which means a comma and a run-on sentence currently
sound identical, with no pause difference at all.

This module fixes that at the AUDIO level instead: split text into chunks
at punctuation marks, synthesize each chunk separately with the existing
normalizer, then stitch the resulting audio together with a silence gap
sized to the punctuation type (short pause for commas, longer for
sentence-ending punctuation).

Usage:
    from kinyarwanda_pause_synthesis import synthesize_with_pauses
    wav, sample_rate = synthesize_with_pauses(text, synthesizer)
"""

import re
import numpy as np

from kinyarwanda_normalizer import normalize_kinyarwanda_text

# Pause duration in milliseconds, by punctuation type. Silence the model adds
# at the start/end of each chunk is TRIMMED first, so these are the real gaps.
# Tune by ear.
PAUSE_DURATIONS_MS = {
    ",": 160,
    ";": 220,
    ":": 220,
    ".": 320,
    "!": 320,
    "?": 360,
    "\n": 420,
}
DEFAULT_PAUSE_MS = 200

# Very short fragments synthesize badly with VITS (odd pronunciation, uneven
# tone, extra silence). A comma-split chunk shorter than this is merged into
# the next one instead (so that comma gets no pause). Sentence-ending marks
# always split.
MIN_WORDS_FOR_COMMA_SPLIT = 4
TRIM_TOP_DB = 35        # silence threshold for trimming each chunk
EDGE_KEEP_MS = 25       # keep a little margin so words are not clipped
FADE_MS = 10

# Optional steadier prosody. VITS defaults are noise_scale=0.667,
# noise_scale_dp=1.0. Lower values = steadier/flatter, higher = more varied.
# Set to None to leave the model's own values untouched.
NOISE_SCALE = None
NOISE_SCALE_DP = None


def split_on_punctuation(text: str) -> list[tuple[str, str]]:
    """Split into (chunk_text, following_punctuation) pairs.
    "Muraho, amakuru? Ni byiza." -> [("Muraho", ","), ("amakuru", "?"), ("Ni byiza", ".")]
    """
    parts = re.split(r"([,;:.!?\n])", text)
    chunks = []
    for i in range(0, len(parts), 2):
        chunk_text = parts[i].strip()
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        if chunk_text:
            chunks.append((chunk_text, punct))
    return chunks


def merge_short_chunks(chunks, min_words=MIN_WORDS_FOR_COMMA_SPLIT):
    """Merge short comma-type fragments into the following chunk."""
    merged, buf = [], ""
    for text, punct in chunks:
        buf = f"{buf} {text}".strip()
        if punct in (",", ";", ":") and len(buf.split()) < min_words:
            continue  # too short to stand alone - keep accumulating
        merged.append((buf, punct))
        buf = ""
    if buf:
        merged.append((buf, ""))
    return merged


def _trim(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    """Cut leading/trailing silence the model adds, keeping a small margin."""
    if wav.size == 0:
        return wav
    peak = float(np.max(np.abs(wav)))
    if peak == 0:
        return wav
    thresh = peak * (10 ** (-TRIM_TOP_DB / 20))
    idx = np.where(np.abs(wav) > thresh)[0]
    if idx.size == 0:
        return wav
    keep = int(sample_rate * EDGE_KEEP_MS / 1000)
    a, b = max(0, idx[0] - keep), min(wav.size, idx[-1] + keep)
    out = wav[a:b].copy()
    f = min(int(sample_rate * FADE_MS / 1000), out.size // 2)
    if f > 0:
        ramp = np.linspace(0, 1, f, dtype=np.float32)
        out[:f] *= ramp
        out[-f:] *= ramp[::-1]
    return out


def _apply_noise(synthesizer):
    model = getattr(synthesizer, "tts_model", None)
    if model is None:
        return
    if NOISE_SCALE is not None and hasattr(model, "inference_noise_scale"):
        model.inference_noise_scale = NOISE_SCALE
    if NOISE_SCALE_DP is not None and hasattr(model, "inference_noise_scale_dp"):
        model.inference_noise_scale_dp = NOISE_SCALE_DP


def synthesize_with_pauses(text: str, synthesizer, sample_rate: int = 22050):
    """Synthesize text with punctuation-appropriate pauses.

    Returns (waveform float32, sample_rate).
    """
    _apply_noise(synthesizer)
    chunks = merge_short_chunks(split_on_punctuation(text))
    if not chunks:
        return np.array([], dtype=np.float32), sample_rate

    pieces = []
    for chunk_text, punct in chunks:
        normalized = normalize_kinyarwanda_text(chunk_text)
        if not normalized:
            continue
        wav = _trim(np.array(synthesizer.tts(normalized), dtype=np.float32), sample_rate)
        pieces.append(wav)
        pause_ms = PAUSE_DURATIONS_MS.get(punct, DEFAULT_PAUSE_MS)
        pieces.append(np.zeros(int(sample_rate * pause_ms / 1000), dtype=np.float32))

    if not pieces:
        return np.array([], dtype=np.float32), sample_rate
    return np.concatenate(pieces[:-1]), sample_rate


if __name__ == "__main__":
    SR = 22050

    class FakeSynthesizer:
        """Returns a tone padded with 500 ms silence each side, like VITS often does."""
        def tts(self, text):
            n = int(SR * 0.1 * len(text.split()))
            tone = 0.5 * np.sin(np.linspace(0, 440 * 2 * np.pi * n / SR, n)).astype(np.float32)
            pad = np.zeros(int(SR * 0.5), dtype=np.float32)
            return np.concatenate([pad, tone, pad])

    t = "Muraho, amakuru? Ni byiza cyane, reka tugende vuba ejo. Yego!"
    print(merge_short_chunks(split_on_punctuation(t)))
    wav, sr = synthesize_with_pauses(t, FakeSynthesizer())
    print(f"{len(wav)/sr:.2f}s total")
