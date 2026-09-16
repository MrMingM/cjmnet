"""Class-balanced point supervision for the two-class evidential head."""

import torch


def evidential_point_loss(output, targets, edm_weight=0.1,
                          wrong_evidence_weight=1.0e-3):
    valid = output["point_valid_mask"] & (targets >= 0)
    target = targets[valid]
    reliability = output["point_reliability"][valid].clamp(1e-5, 1 - 1e-5)
    if target.numel() == 0:
        raise ValueError("point batch contains no supervised slot")
    positive = target >= 0.5
    negative = ~positive
    terms = []
    if positive.any():
        terms.append(-torch.log(reliability[positive]).mean())
    if negative.any():
        terms.append(-torch.log1p(-reliability[negative]).mean())
    balanced_bce = torch.stack(terms).mean()

    evidence = output["point_evidence"][valid]
    alpha = evidence + 1.0
    strength = alpha.sum(dim=-1, keepdim=True)
    probability = alpha / strength
    one_hot = torch.stack((target, 1.0 - target), dim=-1)
    edm = ((one_hot - probability).square() +
           probability * (1.0 - probability) / (strength + 1.0))
    edm = edm.sum(dim=-1).mean()
    wrong = (evidence[:, 0] * (1.0 - target) +
             evidence[:, 1] * target).mean()
    total = balanced_bce + edm_weight * edm + wrong_evidence_weight * wrong
    return total, {
        "balanced_bce": balanced_bce.detach(),
        "edm": edm.detach(),
        "wrong_evidence": wrong.detach(),
        "supervised_points": target.numel(),
        "noise_fraction": negative.float().mean().detach(),
    }


class BinaryHistogramMetrics:
    def __init__(self, bins=1000):
        self.bins = bins
        self.positive = torch.zeros(bins, dtype=torch.double)
        self.negative = torch.zeros(bins, dtype=torch.double)

    def update(self, scores, targets):
        scores = scores.detach().float().cpu().clamp(0, 1)
        targets = targets.detach().float().cpu() >= 0.5
        index = (scores * (self.bins - 1)).long()
        self.positive += torch.bincount(
            index[targets], minlength=self.bins).double()
        self.negative += torch.bincount(
            index[~targets], minlength=self.bins).double()

    def compute(self):
        tp = torch.cumsum(self.positive.flip(0), dim=0)
        fp = torch.cumsum(self.negative.flip(0), dim=0)
        positives = self.positive.sum().clamp_min(1)
        negatives = self.negative.sum().clamp_min(1)
        recall = tp / positives
        false_positive_rate = fp / negatives
        precision = tp / (tp + fp).clamp_min(1)
        zero = torch.zeros(1, dtype=torch.double)
        auroc = torch.trapz(torch.cat((zero, recall)),
                            torch.cat((zero, false_positive_rate)))
        recall_step = recall - torch.cat((zero, recall[:-1]))
        auprc = (recall_step * precision).sum()
        threshold_index = self.bins - 1 - int(0.5 * (self.bins - 1))
        threshold_index = max(0, min(threshold_index, self.bins - 1))
        clean_recall = recall[threshold_index]
        noise_recall = 1.0 - false_positive_rate[threshold_index]
        f1 = (2 * precision[threshold_index] * clean_recall /
              (precision[threshold_index] + clean_recall).clamp_min(1e-12))
        thresholds = torch.arange(
            self.bins - 1, -1, -1, dtype=torch.double) / (self.bins - 1)
        balanced_accuracy = (recall + 1.0 - false_positive_rate) / 2.0
        best_index = int(balanced_accuracy.argmax().item())
        clean95_candidates = torch.nonzero(recall >= 0.95).flatten()
        clean95_index = (int(clean95_candidates[0].item())
                         if clean95_candidates.numel() else self.bins - 1)
        score_centers = torch.arange(self.bins, dtype=torch.double) / \
            (self.bins - 1)
        positive_score_mean = (self.positive * score_centers).sum() / positives
        negative_score_mean = (self.negative * score_centers).sum() / negatives
        return {
            "auroc": auroc.item(), "auprc": auprc.item(),
            "f1_at_0.5": f1.item(),
            "clean_recall_at_0.5": clean_recall.item(),
            "noise_recall_at_0.5": noise_recall.item(),
            "clean_score_mean": positive_score_mean.item(),
            "noise_score_mean": negative_score_mean.item(),
            "best_balanced_threshold": thresholds[best_index].item(),
            "best_balanced_accuracy": balanced_accuracy[best_index].item(),
            "clean_recall_at_best": recall[best_index].item(),
            "noise_recall_at_best":
                (1.0 - false_positive_rate[best_index]).item(),
            "threshold_at_95_clean_recall":
                thresholds[clean95_index].item(),
            "noise_recall_at_95_clean_recall":
                (1.0 - false_positive_rate[clean95_index]).item(),
            "positive_points": int(self.positive.sum().item()),
            "noise_points": int(self.negative.sum().item()),
        }
