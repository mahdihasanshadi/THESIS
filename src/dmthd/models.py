"""Model loading, pooled hidden states, the student with projections and irony head, and the
non-transformer BiLSTM student used as the heterogeneous baseline."""
import json
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

BILSTM_NAME = "bilstm"
BILSTM_TOKENIZER = "bert-base-uncased"   # the BiLSTM reads the same WordPiece ids as the BERT teachers


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """hidden [B, L, d], mask [B, L] -> [B, d]."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)


class BiLSTMClassifier(nn.Module):
    """Heterogeneous student in the sense of Tang et al. (2019): WordPiece ids -> embedding ->
    two-layer BiLSTM -> masked mean pooling -> linear classifier. About 10M parameters with the
    defaults, i.e. the size of BERT-mini, but not a transformer."""

    def __init__(self, vocab_size=30522, num_labels=6, emb_dim=256, hidden=256, layers=2, dropout=0.2, pad_id=0):
        super().__init__()
        self.config = SimpleNamespace(model_type="bilstm", hidden_size=2 * hidden, vocab_size=vocab_size,
                                      num_labels=num_labels, emb_dim=emb_dim, hidden=hidden, layers=layers,
                                      dropout=dropout, pad_id=pad_id)
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden, num_labels)

    def forward_pooled(self, input_ids, attention_mask):
        x = self.drop(self.emb(input_ids))
        lengths = attention_mask.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=input_ids.size(1))
        pooled = masked_mean(out, attention_mask)
        return self.classifier(self.drop(pooled)), pooled

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        logits, pooled = self.forward_pooled(input_ids, attention_mask)
        return SimpleNamespace(logits=logits, pooled=pooled)

    def save_pretrained(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "bilstm_config.json"), "w") as f:
            json.dump(vars(self.config), f)
        torch.save(self.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, d):
        with open(os.path.join(d, "bilstm_config.json")) as f:
            c = json.load(f)
        m = cls(vocab_size=c["vocab_size"], num_labels=c["num_labels"], emb_dim=c["emb_dim"], hidden=c["hidden"],
                layers=c["layers"], dropout=c["dropout"], pad_id=c["pad_id"])
        m.load_state_dict(torch.load(os.path.join(d, "pytorch_model.bin"), map_location="cpu"))
        return m


def is_bilstm(name_or_dir: str) -> bool:
    return name_or_dir == BILSTM_NAME or (os.path.isdir(name_or_dir) and os.path.exists(os.path.join(name_or_dir, "bilstm_config.json")))


def load_tokenizer(name_or_dir: str):
    if name_or_dir == BILSTM_NAME:
        return AutoTokenizer.from_pretrained(BILSTM_TOKENIZER)
    return AutoTokenizer.from_pretrained(name_or_dir)


def load_classifier(name_or_dir: str, num_labels: int):
    """Any HF encoder as a sequence classifier, or a saved BiLSTM student. `ignore_mismatched_sizes`
    lets a checkpoint with another head (e.g. the 2-way irony model) be adapted to this task."""
    if is_bilstm(name_or_dir):
        return BiLSTMClassifier.from_pretrained(name_or_dir) if os.path.isdir(name_or_dir) else BiLSTMClassifier(num_labels=num_labels)
    kw = dict(num_labels=num_labels, ignore_mismatched_sizes=True)
    try:   # transformers >= 5 names the argument `dtype`; older versions `torch_dtype`. Always train in fp32 weights.
        return AutoModelForSequenceClassification.from_pretrained(name_or_dir, dtype=torch.float32, **kw)
    except TypeError:
        return AutoModelForSequenceClassification.from_pretrained(name_or_dir, torch_dtype=torch.float32, **kw)


def encode_batch(model, input_ids, attention_mask):
    """Logits and masked-mean-pooled last-layer state for any classifier."""
    if isinstance(model, BiLSTMClassifier):
        return model.forward_pooled(input_ids, attention_mask)
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    pooled = masked_mean(out.hidden_states[-1], attention_mask)
    return out.logits, pooled


def uses_amp(model_name: str, device, fp16: bool) -> bool:
    """DeBERTa-v3 overflows in fp16 autocast; keep it in fp32."""
    return bool(fp16) and device.type == "cuda" and "deberta" not in model_name.lower()


class Student(nn.Module):
    """Compact classifier + one linear projection per teacher (d_s -> d_k) + optional auxiliary
    irony head (d_s -> aux_labels) on the pooled state."""

    def __init__(self, name: str, num_labels: int, teacher_dims=(), aux_labels: int = 0, from_scratch: bool = False):
        super().__init__()
        if from_scratch and not is_bilstm(name):
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
