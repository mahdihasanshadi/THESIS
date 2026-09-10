"""Model loading, pooled hidden states, and the student with projections and irony head."""
import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """hidden [B, L, d], mask [B, L] -> [B, d]."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)


def load_tokenizer(name_or_dir: str):
    return AutoTokenizer.from_pretrained(name_or_dir)


def load_classifier(name_or_dir: str, num_labels: int):
    """Any HF encoder as a sequence classifier. `ignore_mismatched_sizes` lets a checkpoint
    with another head (e.g. the 2-way irony model) be adapted to this task."""
    return AutoModelForSequenceClassification.from_pretrained(
        name_or_dir, num_labels=num_labels, ignore_mismatched_sizes=True)


def encode_batch(model, input_ids, attention_mask):
    """Logits and masked-mean-pooled last-layer state for any classifier."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    pooled = masked_mean(out.hidden_states[-1], attention_mask)
    return out.logits, pooled


class Student(nn.Module):
    """Compact classifier + one linear projection per teacher (d_s -> d_k) + optional
    auxiliary irony head (d_s -> aux_labels) on the pooled state."""

    def __init__(self, name: str, num_labels: int, teacher_dims=(), aux_labels: int = 0,
                 from_scratch: bool = False):
        super().__init__()
        if from_scratch:
            cfg = AutoConfig.from_pretrained(name, num_labels=num_labels)
            self.base = AutoModelForSequenceClassification.from_config(cfg)
        else:
            self.base = load_classifier(name, num_labels)
        d = self.base.config.hidden_size
        self.proj = nn.ModuleList([nn.Linear(d, int(dk)) for dk in teacher_dims])
        self.aux = nn.Linear(d, aux_labels) if aux_labels else None

    def forward(self, input_ids, attention_mask):
        logits, pooled = encode_batch(self.base, input_ids, attention_mask)
        aux = self.aux(pooled) if self.aux is not None else None
        return logits, pooled, aux

    def save(self, out_dir: str, tokenizer=None):
        os.makedirs(out_dir, exist_ok=True)
        self.base.save_pretrained(out_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(out_dir)
        heads = {"proj": self.proj.state_dict(), "aux": self.aux.state_dict() if self.aux is not None else None}
        torch.save(heads, os.path.join(out_dir, "dmthd_heads.pt"))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
