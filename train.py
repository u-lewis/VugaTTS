
"""
VITS training script for Kinyarwanda single-speaker TTS.
Based on Coqui-TTS (idiap/coqui-ai-TTS fork, package name: coqui-tts) recipes.

RESUME STRATEGY (important):
Uses `continue_path`, NOT `restore_path`, to resume. This is a deliberate
fix - restore_path only reloads model WEIGHTS and discards the optimizer's
learned momentum/variance estimates (Adam's internal state) every time,
forcing the optimizer to "re-warm" from scratch on every restart. This is
wasted compute and slows convergence. continue_path restores the full
training state - optimizer, LR scheduler, and step/epoch counters - and
writes new checkpoints into the SAME run folder instead of spawning a new
one each time.

This script auto-detects the most recently modified run folder under
RUN_OUTPUT_PATH and passes it as continue_path automatically - no manual
path editing needed between sessions. If no run folder exists yet, it
starts fresh.
"""

import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits

# Version-proof imports (newer coqui-tts moved these classes)
try:
    from TTS.config.shared_configs import BaseDatasetConfig
except ImportError:
    from TTS.tts.configs.shared_configs import BaseDatasetConfig
try:
    from TTS.tts.models.vits import VitsAudioConfig
except ImportError:
    from TTS.config.shared_configs import BaseAudioConfig as VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from kinyarwanda_formatter import kinyarwanda_formatter
from checkpoint_utils import find_best_valid_checkpoint

OUTPUT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(OUTPUT_PATH, "data")
RUN_OUTPUT_PATH = "/content/drive/MyDrive/VugaTTS_runs"

# ---------------------------------------------------------------------------
# Robust resume: scans ALL run folders, verifies checkpoints actually load
# (not just that the filename looks right), and picks the highest-step
# VALID one. Automatically falls back past corrupted files - Drive/Colab
# disconnects mid-write have corrupted checkpoints twice in this project,
# and this makes that a non-issue going forward instead of a manual fix.
# ---------------------------------------------------------------------------
print("Scanning for the best valid checkpoint to resume from...")
CONTINUE_PATH, _, _ = find_best_valid_checkpoint(RUN_OUTPUT_PATH)

if CONTINUE_PATH:
    print(f"\nWill CONTINUE (optimizer + scheduler + step state restored) from: {CONTINUE_PATH}")
else:
    print("\nNo valid run folder found - starting a fresh training run.")

KINYARWANDA_CHARACTERS = CharactersConfig(
    characters="abcdefghijklmnopqrstuvwyz ",  # includes space explicitly
    punctuations="'",
    pad="<PAD>",
    eos="<EOS>",
    bos="<BOS>",
    blank="<BLNK>",
)

dataset_config = BaseDatasetConfig(
    formatter="kinyarwanda",
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
    batch_size=16,
    eval_batch_size=8,
    batch_group_size=5,
    num_loader_workers=4,
    num_eval_loader_workers=2,
    run_eval=True,
    test_delay_epochs=0,
    epochs=1000,
    text_cleaner="basic_cleaners",
    use_phonemes=False,
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    output_path=RUN_OUTPUT_PATH,
    datasets=[dataset_config],
    characters=KINYARWANDA_CHARACTERS,
    save_step=500,
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
tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_size=0.02,
    formatter=kinyarwanda_formatter,
)

model = Vits(config, ap, tokenizer)

trainer_args = TrainerArgs(continue_path=CONTINUE_PATH) if CONTINUE_PATH else TrainerArgs()

trainer = Trainer(
    trainer_args,
    config,
    RUN_OUTPUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

# --- Diagnostic: confirm optimizer state actually carried over ---
# If this is a genuine continue (not a fresh start), the optimizer should
# NOT be at its initial "cold" state. We can't directly inspect Adam's
# internal moment estimates cheaply here, but we can confirm the restored
# step count and LR match what's expected instead of resetting to step 0/LR max.
try:
    restored_step = getattr(trainer, "total_steps_done", None)
    print(f"[Resume check] trainer.total_steps_done after init: {restored_step}")
except Exception as e:
    print(f"[Resume check] Could not read step counter directly: {e}")

trainer.fit()
