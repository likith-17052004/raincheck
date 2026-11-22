# GPU runbook (Nebius H100, about $2.20/hr)

Stop the VM when a step is done. Budget: $100 is about 45 GPU hours; this plan needs 4-8.

1. **Start the VM** in the console and copy its public IP (it may have changed).
2. **Sync files** (from your Mac): `./training/sync_to_vm.sh <user> <ip>`
3. **Set up** on the VM: `ssh <user>@<ip>` then `bash ~/forecasttrust/training/setup_vm.sh`
   Installs dependencies, runs the CPU unit test and the baselines on the TimeNet data.
4. **Hugging Face login, done by you on the VM** (never paste the token in chat):
   `source ~/forecasttrust/.venv/bin/activate && hf auth login`
   (older versions: `huggingface-cli login`). The Llama-3.2-1B repo needs access approval first.
5. **Smoke test** (about 5 min), from `~/forecasttrust/training`:
   `python train_forecasttrust.py --max-train 64 --epochs 1 --out ../results/smoke`
6. **Main run** (est. 0.5-1.5 h): `python train_forecasttrust.py --out ../results/main_s0`
7. **Ablation, no verified-rain series**: `python train_forecasttrust.py --no-history --out ../results/nohist_s0`
8. **More seeds** (`--seed 1`, `--seed 2`) if budget allows, then copy `results/` back and **stop the VM**.

If Llama access is not approved yet, use an ungated base: `--llm Qwen/Qwen2.5-1.5B`
(unverified with OpenTSLM-SP; the smoke test shows whether it works).
