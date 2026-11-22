"""Fine-tune OpenTSLM-SP (frozen LLM + LoRA, trainable encoder and projector) on ForecastTrust.

Samples come from the TimeNet dataset forecasttrust/india-rainfall-reliability with its stored
chronological splits. The model answers "reliable" or "failure". The score used for AUROC, average
precision and Brier is P(failure) = sigmoid(logP("failure") - logP("reliable")), computed with the
same teacher-forced format as training, so no free-text parsing is involved.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.nn.utils.rnn import pad_sequence

from forecasttrust_data import LABELS, Row, load_rows

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]  # fmt: skip


def to_sample(row: Row, *, history: bool) -> dict:
    """Turn a row into the prompt dict that OpenTSLM-SP expects."""
    f1, f2, f3 = row.fc1[-1], row.fc2[-1], row.fc3[-1]
    pre = (
        "You assess whether a day-ahead rainfall forecast will fail before the rain is observed. "
        f"City: {row.city}. Valid day: {row.valid_date.isoformat()} ({MONTHS[row.valid_date.month - 1]}). "
        f"Forecast rainfall for the valid day: {f1:.1f} mm from the day-1 run, {f2:.1f} mm from the day-2 run, "
        f"{f3:.1f} mm from the day-3 run. A failure means the day-1 forecast error exceeds the larger of "
        "5 mm and half the forecast."
    )
    series = [
        ("Day-1 forecasts for the 14 days ending on the valid day, log(1+mm).", row.fc1),
        ("Day-2 forecasts for the same days, log(1+mm).", row.fc2),
        ("Day-3 forecasts for the same days, log(1+mm).", row.fc3),
    ]
    if history:
        series.append(("Verified rainfall for the 12 days ending two days before the valid day, log(1+mm).", row.observed))
    return {
        "pre_prompt": pre,
        "time_series_text": [text for text, _ in series],
        "time_series": [np.log1p(np.clip(values, 0, None)).astype(np.float32) for _, values in series],
        "post_prompt": "Answer with one word, reliable or failure.",
        "label": row.label,
    }


def collate(batch: list[dict]) -> list[dict]:
    """Pad every sample's series to a multiple of the patch size, as OpenTSLM requires."""
    from opentslm.model_config import PATCH_SIZE
    from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

    return extend_time_series_to_match_patch_size_and_aggregate(batch, patch_size=PATCH_SIZE)


def encode(model, batch: list[dict], words: list[str]):
    """Concatenate each sample's prompt embeddings with its answer tokens, with no padding in between."""
    prompt_embeds, prompt_mask = model.pad_and_apply_batch(batch)
    embed = model.llm.get_input_embeddings()
    device = prompt_embeds.device
    seqs, labels = [], []
    for i, word in enumerate(words):
        n = int(prompt_mask[i].sum().item())
        ids = model.tokenizer(word, add_special_tokens=False).input_ids + [model.tokenizer.eos_token_id]
        ids_t = torch.tensor(ids, device=device)
        seqs.append(torch.cat([prompt_embeds[i, :n], embed(ids_t).to(prompt_embeds.dtype)], dim=0))
        labels.append(torch.cat([torch.full((n,), -100, device=device), ids_t]))
    lengths = torch.tensor([s.size(0) for s in seqs], device=device)
    x = pad_sequence(seqs, batch_first=True)
    y = pad_sequence(labels, batch_first=True, padding_value=-100)
    mask = (torch.arange(x.size(1), device=device)[None, :] < lengths[:, None]).long()
    return x, y, mask


def answer_logprob(model, batch: list[dict], words: list[str]) -> torch.Tensor:
    """Per-sample log-probability of each sample's answer (and end token), plus the token count."""
    x, y, mask = encode(model, batch, words)
    logits = model.llm(inputs_embeds=x, attention_mask=mask).logits[:, :-1].float()
    target = y[:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * (target != -100)
    return token_logp.sum(dim=1), (target != -100).sum(dim=1)


@torch.no_grad()
def score(model, samples: list[dict], batch_size: int) -> np.ndarray:
    """Return the logit of P(failure) for every sample."""
    model.eval()
    out = []
    for start in range(0, len(samples), batch_size):
        batch = collate([dict(s) for s in samples[start : start + batch_size]])
        lp_fail, _ = answer_logprob(model, batch, ["failure"] * len(batch))
        lp_ok, _ = answer_logprob(model, batch, ["reliable"] * len(batch))
        out.append((lp_fail - lp_ok).cpu().numpy())
    return np.concatenate(out)


def metrics(y: np.ndarray, logit: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """AUROC and average precision of the raw score, and Brier of the given probabilities."""
    return {"auroc": float(roc_auc_score(y, logit)), "ap": float(average_precision_score(y, logit)), "brier": float(brier_score_loss(y, prob))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(Path(__file__).resolve().parent.parent / "registry"))
    parser.add_argument("--llm", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-encoder", type=float, default=2e-4)
    parser.add_argument("--lr-projector", type=float, default=1e-4)
    parser.add_argument("--lr-lora", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--no-history", action="store_true", help="Ablation: drop the verified-rainfall series.")
    parser.add_argument("--max-train", type=int, default=None, help="Smoke test: limit samples per split.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/opentslm_run")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.registry)
    data = {k: [to_sample(r, history=not args.no_history) for r in v] for k, v in rows.items()}
    if args.max_train:
        rng = random.Random(args.seed)
        data = {k: rng.sample(v, min(args.max_train, len(v))) for k, v in data.items()}
    print({k: (len(v), round(float(np.mean([s["label"] for s in v])), 3)) for k, v in data.items()}, flush=True)

    from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OpenTSLMSP(llm_id=args.llm, device=device)
    model.enable_lora(lora_r=args.lora_r, lora_alpha=2 * args.lora_r)
    groups = [
        {"params": list(model.encoder.parameters()), "lr": args.lr_encoder},
        {"params": list(model.projector.parameters()), "lr": args.lr_projector},
        {"params": model.get_lora_parameters(), "lr": args.lr_lora},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-2)
    steps = args.epochs * ((len(data["train"]) + args.batch_size - 1) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.1, anneal_strategy="cos"
    )
    trainable = [p for g in groups for p in g["params"]]
    best_path = out_dir / "best.pt"
    best_auc, bad_epochs, history = -1.0, 0, []
    y_val = np.array([s["label"] for s in data["validation"]])
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(len(data["train"]))
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = collate([dict(data["train"][i]) for i in order[start : start + args.batch_size]])
            words = [LABELS[s["label"]] for s in batch]
            lp, n_tokens = answer_logprob(model, batch, words)
            loss = -(lp.sum() / n_tokens.sum())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        val_logit = score(model, data["validation"], args.batch_size)
        val_auc = float(roc_auc_score(y_val, val_logit)) if len(set(y_val)) > 1 else float("nan")
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_auroc": val_auc, "elapsed_s": time.time() - start_time})
        print(history[-1], flush=True)
        if not best_path.exists() or val_auc > best_auc:
            best_auc, bad_epochs = val_auc, 0
            model.store_to_file(str(best_path))
        else:
            bad_epochs += 1
            if bad_epochs > args.patience:
                break

    model.load_from_file(str(best_path))
    val_logit = score(model, data["validation"], args.batch_size)
    test_logit = score(model, data["test"], args.batch_size)
    y_test = np.array([s["label"] for s in data["test"]])
    raw_prob = 1 / (1 + np.exp(-test_logit))
    # Platt scaling fitted on validation only; it changes calibration, never the ranking.
    platt = LogisticRegression().fit(val_logit[:, None], y_val)
    cal_prob = platt.predict_proba(test_logit[:, None])[:, 1]
    result = {
        "args": vars(args),
        "test_n": int(len(y_test)),
        "test_prevalence": float(y_test.mean()),
        "raw": metrics(y_test, test_logit, raw_prob),
        "platt_on_validation": metrics(y_test, test_logit, cal_prob),
        "best_val_auroc": best_auc,
        "history": history,
        "gpu_seconds": time.time() - start_time,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=1))
    np.save(out_dir / "test_logit.npy", test_logit)
    print(json.dumps({k: result[k] for k in ("raw", "platt_on_validation", "gpu_seconds")}, indent=1))


if __name__ == "__main__":
    main()
