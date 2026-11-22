"""CPU test of the batching and scoring logic with a stand-in model (no LLM download needed)."""

import sys
import types

import numpy as np
import torch
from torch import nn

import train_forecasttrust as t
from forecasttrust_data import load_rows

VOCAB, HID, EOS = 50, 16, 1


class StubTokenizer:
    eos_token_id = EOS

    def __call__(self, word, add_special_tokens=False):
        ids = [2 + (ord(c) % 40) for c in word]
        return types.SimpleNamespace(input_ids=ids)


class StubLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, HID)
        self.rnn = nn.GRU(HID, HID, batch_first=True)
        self.head = nn.Linear(HID, VOCAB)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, inputs_embeds, attention_mask):
        hidden, _ = self.rnn(inputs_embeds * attention_mask[..., None])
        return types.SimpleNamespace(logits=self.head(hidden))


class StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = StubLLM()
        self.tokenizer = StubTokenizer()
        self.proj = nn.Linear(1, HID)

    def pad_and_apply_batch(self, batch):
        seqs = []
        for sample in batch:
            n_text = 5 + len(sample["pre_prompt"]) % 7  # prompts differ in length, like real text
            generator = torch.Generator().manual_seed(len(sample["pre_prompt"]) * 7919)
            text = torch.randn(n_text, HID, generator=generator)  # deterministic per prompt
            ts = self.proj(sample["time_series"].reshape(-1, 1)[::4])
            seqs.append(torch.cat([text, ts], 0))
        lengths = torch.tensor([s.size(0) for s in seqs])
        x = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        mask = (torch.arange(x.size(1))[None] < lengths[:, None]).long()
        return x, mask


def _fake_collate(batch):
    # Mirrors opentslm's extend_time_series_to_match_patch_size_and_aggregate (patch size 4).
    for element in batch:
        tensors = [torch.as_tensor(a, dtype=torch.float32) for a in element["time_series"]]
        padded_len = -(-max(x.size(0) for x in tensors) // 4) * 4
        element["time_series"] = torch.stack([nn.functional.pad(x, (0, padded_len - x.size(0))) for x in tensors])
    return batch


def test_loss_and_scores_are_finite_and_trainable(monkeypatch):
    monkeypatch.setattr(t, "collate", _fake_collate)
    rows = load_rows("../registry")["test"][:48]
    samples = [t.to_sample(r, history=True) for r in rows]
    assert len(samples[0]["time_series"]) == 4 and len(samples[0]["time_series"][3]) == 12
    assert len(t.to_sample(rows[0], history=False)["time_series"]) == 3
    model = StubModel()
    batch = _fake_collate([dict(s) for s in samples[:8]])
    words = [t.LABELS[s["label"]] for s in batch]
    lp, n = t.answer_logprob(model, batch, words)
    assert lp.shape == (8,) and torch.isfinite(lp).all() and (n == torch.tensor([len(w) + 1 for w in words])).all()
    loss = -(lp.sum() / n.sum())
    loss.backward()
    assert model.llm.head.weight.grad is not None
    logits = t.score(model, samples, batch_size=16)
    assert logits.shape == (48,) and np.isfinite(logits).all()
    # Batch size must not change a sample's score (no cross-sample leakage through padding).
    again = t.score(model, samples, batch_size=5)
    np.testing.assert_allclose(logits, again, atol=1e-5)
