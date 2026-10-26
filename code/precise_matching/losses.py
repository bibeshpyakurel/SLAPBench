"""Balanced same-hand cosine verification objective for reviewed training batches."""

from .data import InputError


def verification_loss(embeddings, subjects, captures, hands, margin=0.2):
    import torch

    count = len(subjects)
    if count < 4 or len(captures) != count or len(hands) != count:
        raise InputError("A verification batch needs aligned subject/capture/hand records.")
    if not all(subjects) or not all(captures) or len(set(captures)) != count:
        raise InputError("Training images must have distinct reviewed captures.")
    if len(set(hands)) != 1 or hands[0] not in {"left", "right"}:
        raise InputError("Training microbatches must use one four-finger hand type.")
    if not -1 <= margin <= 1 or embeddings.ndim != 2 or embeddings.shape[0] != count:
        raise InputError("Invalid verification objective inputs.")
    vectors = embeddings.float()
    norms = torch.linalg.vector_norm(vectors, dim=1)
    if not torch.isfinite(vectors).all() or not torch.allclose(
            norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
        raise InputError("Verification loss requires finite normalized embeddings.")
    pairs = torch.triu_indices(count, count, offset=1, device=vectors.device)
    same = torch.tensor([subjects[a] == subjects[b] for a, b in pairs.T.tolist()],
                        dtype=torch.bool, device=vectors.device)
    if not same.any() or same.all():
        raise InputError("Verification batch needs both positive and negative pairs.")
    scores = (vectors[pairs[0]] * vectors[pairs[1]]).sum(dim=1)
    loss = 0.5 * (1 - scores[same]).mean() + 0.5 * (scores[~same] - margin).clamp_min(0).mean()
    return loss, {"positive_pairs": int(same.sum()), "negative_pairs": int((~same).sum())}
