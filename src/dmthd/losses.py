"""The D-MTHD objective, per instance i, committee teachers k = 1..K.

    w_k(i)  = softmax_k( -l_k(i) / tau ),  l_k(i) = CE(p_k(i), y_i)
    p_bar(i) = sum_k w_k(i) softmax(z_k(i)/T)
    L_KL    = T^2 KL( p_bar(i) || softmax(z_s(i)/T) )
    L_hid   = sum_k w_k(i) || W_k h_s(i) - h_k(i) ||^2
    L_soft  = BCE( sigma(z_s^attack(i)), s_i )            (only with annotator fractions)
    L_irony = T^2 KL( softmax(z_irony(i)/T) || softmax(z_s^irony(i)/T) )
    L = alpha L_KL + beta CE + gamma (0.5 L_hid + 0.5 L_soft) + delta L_irony
"""
import torch
import torch.nn.functional as F


def teacher_weights(teacher_logits: torch.Tensor, labels: torch.Tensor, tau: float,
                    per_instance: bool = True, uniform: bool = False) -> torch.Tensor:
    """teacher_logits [K, B, C] -> weights [B, K]."""
    K, B, _ = teacher_logits.shape
    if uniform:
        return torch.full((B, K), 1.0 / K, device=teacher_logits.device)
    err = torch.stack([F.cross_entropy(teacher_logits[k], labels, reduction="none") for k in range(K)], dim=1)
    if not per_instance:
        err = err.mean(0, keepdim=True).expand(B, K)
    return F.softmax(-err / tau, dim=1)


def kd_kl(student_logits: torch.Tensor, target_probs: torch.Tensor, T: float) -> torch.Tensor:
    """T^2 * KL(target || student) with the standard temperature scaling."""
    return F.kl_div(F.log_softmax(student_logits / T, dim=-1), target_probs, reduction="batchmean") * (T * T)


def dmthd_loss(student_logits, labels, *, teacher_logits=None, T=4.0, tau=1.0, alpha=0.4, beta=0.4,
               gamma=0.2, per_instance=True, uniform=False, student_pooled=None, teacher_pooled=None,
               projections=None, use_hidden=True, soft_targets=None, aux_logits=None,
               aux_teacher_logits=None, delta=0.0, class_weights=None):
    """Returns (total, parts dict, mean teacher weights [K] or None).

    teacher_logits : [K, B, C] cached logits of the committee (None for fine-tune only)
    teacher_pooled : list of K tensors [B, d_k]
    projections    : nn.ModuleList of K Linear(d_s, d_k)
    """
    dev = student_logits.device
    parts = {}
    ce = F.cross_entropy(student_logits, labels, weight=class_weights)
    parts["ce"] = ce
    if teacher_logits is None:
        return ce, parts, None

    w = teacher_weights(teacher_logits, labels, tau, per_instance=per_instance, uniform=uniform)  # [B, K]
    tprobs = F.softmax(teacher_logits / T, dim=-1)                                                  # [K, B, C]
    ensemble = (w.t().unsqueeze(-1) * tprobs).sum(0)                                                # [B, C]
    kl = kd_kl(student_logits, ensemble, T)
    parts["kl"] = kl

    hid = torch.zeros((), device=dev)
    if use_hidden and teacher_pooled is not None and projections is not None:
        for k, proj in enumerate(projections):
            mse = ((proj(student_pooled) - teacher_pooled[k]) ** 2).mean(-1)                        # [B]
            hid = hid + (w[:, k] * mse).mean()
    parts["hidden"] = hid

    soft = torch.zeros((), device=dev)
    if soft_targets is not None:
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
