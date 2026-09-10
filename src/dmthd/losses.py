"""The D-MTHD objective, per instance i, committee teachers k = 1..K.

    w_k(i)   = softmax_k( -l_k(i) / tau ),  l_k(i) = CE(p_k(i), y_i)      (reliability = "hard")
                                            l_k(i) = BCE(p_k^attack(i), s_i) (reliability = "soft", binary tasks)
    p_bar(i) = sum_k w_k(i) softmax(z_k(i)/T)
    L_KL(i)  = T^2 KL( p_bar(i) || softmax(z_s(i)/T) )
    L_hid    = sum_k w_k(i) || W_k h_s(i) - h_k(i) ||^2
    L_soft   = BCE( sigma(z_s^attack(i)), s_i )              (only with annotator fractions)
    L_irony  = T^2 KL( softmax(z_irony(i)/T) || softmax(z_s^irony(i)/T) )

    L = alpha L_KL + beta CE + gamma (0.5 L_hid + 0.5 L_soft) + delta L_irony

Disagreement-aware variant (annotator fraction s_i available): a_i = 1 - H_b(s_i)/ln 2 is the
agreement of the annotators. The hard-label term is scaled by a_i and the teacher term by
1 + kappa (1 - a_i), so the student trusts the committee more exactly where the annotators
disagreed.
"""
import math

import torch
import torch.nn.functional as F


def binary_entropy(s: torch.Tensor) -> torch.Tensor:
    s = s.clamp(1e-6, 1 - 1e-6)
    return -(s * torch.log(s) + (1 - s) * torch.log(1 - s))


def agreement_from_soft(s: torch.Tensor) -> torch.Tensor:
    """a_i in [0, 1]: 1 when all annotators agree, 0 at a 50/50 split."""
    return 1.0 - binary_entropy(s) / math.log(2.0)


def teacher_weights(teacher_logits: torch.Tensor, labels: torch.Tensor, tau: float, per_instance: bool = True,
                    uniform: bool = False, soft_targets: torch.Tensor = None) -> torch.Tensor:
    """teacher_logits [K, B, C] -> weights [B, K]. With soft_targets (binary tasks), reliability is
    measured against the annotator fraction instead of the majority label."""
    K, B, C = teacher_logits.shape
    if uniform:
        return torch.full((B, K), 1.0 / K, device=teacher_logits.device)
    if soft_targets is not None and C == 2:
        p1 = F.softmax(teacher_logits, dim=-1)[..., 1].clamp(1e-6, 1 - 1e-6)          # [K, B]
        s = soft_targets.unsqueeze(0).expand_as(p1).to(p1.dtype)
        err = F.binary_cross_entropy(p1, s, reduction="none").t()                   # [B, K]
    else:
        err = torch.stack([F.cross_entropy(teacher_logits[k], labels, reduction="none") for k in range(K)], dim=1)
    if not per_instance:
        err = err.mean(0, keepdim=True).expand(B, K)
    return F.softmax(-err / tau, dim=1)


def kd_kl_per_instance(student_logits: torch.Tensor, target_probs: torch.Tensor, T: float) -> torch.Tensor:
    """[B]: T^2 * KL(target || student) for each instance."""
    logq = F.log_softmax(student_logits / T, dim=-1)
    p = target_probs.clamp_min(1e-8)
    return (p * (torch.log(p) - logq)).sum(-1) * (T * T)


def kd_kl(student_logits: torch.Tensor, target_probs: torch.Tensor, T: float) -> torch.Tensor:
    return kd_kl_per_instance(student_logits, target_probs, T).mean()


def dmthd_loss(student_logits, labels, *, teacher_logits=None, T=4.0, tau=1.0, alpha=0.4, beta=0.4,
               gamma=0.2, per_instance=True, uniform=False, student_pooled=None, teacher_pooled=None,
               projections=None, use_hidden=True, soft_targets=None, aux_logits=None,
               aux_teacher_logits=None, delta=0.0, class_weights=None, agreement=None, kappa=1.0,
               reliability="hard"):
    """Returns (total, parts dict, mean teacher weights [K] or None).

    teacher_logits : [K, B, C] cached logits of the committee (None for fine-tune only)
    teacher_pooled : list of K tensors [B, d_k]
    projections    : nn.ModuleList of K Linear(d_s, d_k)
    agreement      : [B] annotator agreement a_i (disagreement-aware variant) or None
    """
    dev = student_logits.device
    parts = {}
    ce_i = F.cross_entropy(student_logits, labels, weight=class_weights, reduction="none")
    ce = (agreement * ce_i).mean() if agreement is not None else ce_i.mean()
    parts["ce"] = ce
    if teacher_logits is None:
        return ce, parts, None

    w = teacher_weights(teacher_logits, labels, tau, per_instance=per_instance, uniform=uniform,
                        soft_targets=soft_targets if reliability == "soft" else None)            # [B, K]
    tprobs = F.softmax(teacher_logits / T, dim=-1)                                                  # [K, B, C]
    ensemble = (w.t().unsqueeze(-1) * tprobs).sum(0)                                                # [B, C]
    kl_i = kd_kl_per_instance(student_logits, ensemble, T)
    kl = ((1.0 + kappa * (1.0 - agreement)) * kl_i).mean() if agreement is not None else kl_i.mean()
    parts["kl"] = kl

    hid = torch.zeros((), device=dev)
    if use_hidden and teacher_pooled is not None and projections is not None:
        for k, proj in enumerate(projections):
            mse = ((proj(student_pooled) - teacher_pooled[k]) ** 2).mean(-1)                        # [B]
            hid = hid + (w[:, k] * mse).mean()
    parts["hidden"] = hid

    soft = torch.zeros((), device=dev)
    if soft_targets is not None and student_logits.shape[-1] == 2:
        p_attack = F.softmax(student_logits, dim=-1)[:, 1].clamp(1e-6, 1 - 1e-6)
        soft = F.binary_cross_entropy(p_attack, soft_targets.to(p_attack.dtype))
    parts["soft"] = soft
    feat = 0.5 * hid + 0.5 * soft if soft_targets is not None else hid

    total = alpha * kl + beta * ce + gamma * feat
    if aux_logits is not None and aux_teacher_logits is not None and delta > 0:
        aux = kd_kl(aux_logits, F.softmax(aux_teacher_logits / T, dim=-1), T)
        parts["aux"] = aux
        total = total + delta * aux
    return total, parts, w.detach().mean(0)
