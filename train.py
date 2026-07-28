"""
VITS training script for Kinyarwanda single-speaker TTS.
Based on Coqui-TTS (idiap/coqui-ai-TTS fork, package name: coqui-tts) recipes.

Expected directory layout (adjust paths below if different):
  data/processed/wavs_22050/*.wav
  data/train_filelist_22050.txt   (path|text, pipe-separated)
  data/val_filelist_22050.txt

Run:
  python train_vits.py
"""

import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.characters import BaseCharacters
from TTS.utils.audio import AudioProcessor

from kinyarwanda_formatter import kinyarwanda_formatter

# ---------------------------------------------------------------------------
# Paths - adjust these to match your environment (Lightning AI storage, etc.)
# ---------------------------------------------------------------------------
OUTPUT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(OUTPUT_PATH, "data")
WAVS_PATH = os.path.join(DATA_PATH, "processed", "wavs_22050")
TRAIN_FILELIST = os.path.join(DATA_PATH, "train_filelist_22050.txt")
VAL_FILELIST = os.path.join(DATA_PATH, "val_filelist_22050.txt")
RUN_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "runs")

# ---------------------------------------------------------------------------
# Kinyarwanda character set - derived from actual dataset transcripts
# (27 unique characters: space, apostrophe, and 25 letters, no punctuation)
# ---------------------------------------------------------------------------
KINYARWANDA_CHARACTERS = BaseCharacters(
    characters="abcdefghijklmnopqrstuvwyz",  # letters seen in transcripts (no 'x')
    punctuations="'",
    pad="<PAD>",
    eos="<EOS>",
    bos="<BOS>",
    blank="<BLNK>",
)

# ---------------------------------------------------------------------------
# Dataset config - "formatter" name here is unused when a formatter callable
# is passed directly to load_tts_samples() below, but Coqui still requires
# the field to be set. meta_file_train/val point at our path|text filelists,
# read by kinyarwanda_formatter().
# ---------------------------------------------------------------------------
dataset_config = BaseDatasetConfig(
    formatter="kinyarwanda",  # informational only; actual parsing uses kinyarwanda_formatter
    meta_file_train="train_filelist_22050.txt",
    meta_file_val="val_filelist_22050.txt",
    path=DATA_PATH,
)

audio_config = VitsAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None,
)

config = VitsConfig(
    audio=audio_config,
    run_name="kinyarwanda_vits",
    batch_size=16,          # lower if you hit OOM; raise if GPU has headroom (e.g. A100)
    eval_batch_size=8,
    batch_group_size=5,
    num_loader_workers=4,
    num_eval_loader_workers=2,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="basic_cleaners",  # no phonemizer needed - character-based input
    use_phonemes=False,
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    output_path=RUN_OUTPUT_PATH,
    datasets=[dataset_config],
    characters=KINYARWANDA_CHARACTERS,
    save_step=1000,
    save_n_checkpoints=3,
    save_best_after=1000,
    target_loss="loss_1",
    cudnn_benchmark=True,
    test_sentences=[
        "amaze kubona izi mbwa ngo yageze i kigali",
        "iyo nzu ni nto ariko ni nini bihagije kuri twe",
    ],
)

ap = AudioProcessor.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_size=0.02,
    formatter=kinyarwanda_formatter,
)

model = Vits(config, ap)

trainer = Trainer(
    TrainerArgs(),
    config,
    RUN_OUTPUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()
