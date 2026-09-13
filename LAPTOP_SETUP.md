# Moving this project to another machine

Everything the project needs lives on the external drive — the repo, the corpus, the
runs, the venv, the HuggingFace cache. But four things live *outside* it on the machine
it was set up on, and each one fails in a confusing way rather than an obvious one.

## 1. The drive must mount as `G:`

**14 files hardcode `G:`** — `configs/audio.yaml`, `configs/default.yaml`, `score_run.py`,
`build_playtest.py`, `measure_pattern_spans.py`, `separate_stems.py`, the manifest
builders, and `dataloader/utils_dataloader.py` among them.

Disk Management → right-click the volume → Change Drive Letter and Paths → `G:`.

Thirty seconds, and it avoids editing fourteen files. If it mounts as `E:` or `F:`
nothing works and the errors point at missing files rather than at the drive letter.

## 2. `HF_HOME` must point at the cache on the drive

    setx HF_HOME "G:\a2c_data\hf_cache"

Then open a new terminal, because `setx` does not affect the one it ran in.

This is a Windows *user* environment variable, not anything in the code, so it does not
travel with the drive. Without it, HuggingFace ignores the 1.8 GB of models already
sitting in `G:\a2c_data\hf_cache` and re-downloads Encodec and the charter checkpoints to
the laptop's C: drive.

Verify: `python -c "import os; print(os.environ.get('HF_HOME'))"`

## 3. Python 3.13 must be installed

`.venv313\Scripts\python.exe` needs a base Python 3.13 to run against. Note that the
venv's `pyvenv.cfg` already points at `C:\Users\rober\AppData\Local\Programs\Python\Python313`,
which does not exist on the machine this was set up on either — it works because a 3.13 is
installed *somewhere* it can find.

If the venv will not start on the new machine, rebuild it rather than fight it:

    py -3.13 -m venv G:\shuff\audio2chart\.venv313
    G:\shuff\audio2chart\.venv313\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    G:\shuff\audio2chart\.venv313\Scripts\python.exe -m pip install lightning hydra-core omegaconf transformers torchmetrics librosa soundfile wandb demucs

## 4. NVIDIA driver new enough for CUDA 12.8

    .venv313\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

Must print `True` and name the GPU. If it prints `False`, training is off the table —
CPU training is roughly a hundred times slower, which is not a smaller version of the
same thing, it is a different project.

## Then check it actually works

    .venv313\Scripts\python.exe -m unittest discover -s tests -t .

319 tests. If those pass, the corpus, the catalogue and the tokenizer are all reachable.

Then a two-minute training smoke test before committing to anything long:

    .venv313\Scripts\python.exe main.py --config-name audio ^
      data.split_folder=G:/a2c_data/splits_tapping loader.train_num_pieces=1 ^
      trainer.accumulate_grad_batches=2 trainer.max_steps=4 ^
      trainer.val_check_interval=4 trainer.limit_val_batches=3 ^
      trainer.save_run=false tracking.enabled=false

## Sizing for a different GPU

The config assumes **8 GB of VRAM** and uses about 7 GB of it. Same on an RTX 3060 Ti and
an RTX 4060, so it transfers as-is between those two.

With less VRAM, in order of what to give up first:

- `model.freeze_layers=8` (from 4) — fewer gradients and optimizer states
- `loader.num_workers=4` (from 6) — less host RAM behind the loader
- `loader.val_num_pieces=1` (from 4) — smaller validation batches, at the cost of a
  noisier monitored metric

Watch for the failure mode rather than an error: Windows WDDM silently spills VRAM to
host RAM instead of raising, so memory pressure shows up as a roughly 20x slowdown and
never as an out-of-memory. If throughput collapses, check VRAM before anything else.

And note that `nvidia-smi` utilization lies about this — it reports whether any kernel
ran in a sampling window, not how busy the card is. Check `power.draw` and `clocks.sm`
too; a card reading 100% utilization at 74 W of a 265 W limit is idling, not working.

## Unplugging safely

Checkpoints are 2.42 GB. Stop every Python process and confirm the GPU is released before
ejecting — a drive pulled mid-write leaves a truncated checkpoint, and the error it
produces later points at a corrupt zip rather than at the unplug.

    Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' } | Stop-Process -Force
    nvidia-smi --query-gpu=memory.used --format=csv,noheader

Then eject through Windows rather than pulling the cable.
