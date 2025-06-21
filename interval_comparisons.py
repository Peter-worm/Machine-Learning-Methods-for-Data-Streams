from typing import List, Tuple
import numpy as np

def interval_overlap(a, b) -> int:
    """Length of the overlap between two intervals."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def interval_union(a, b) -> int:
    """Length of the union between two intervals."""
    return max(a[1], b[1]) - min(a[0], b[0])

def iou(a, b) -> float:
    inter = interval_overlap(a, b)
    union = interval_union(a, b)
    return inter / union if union > 0 else 0.0

def greedy_match(predicted: List, ground_truth: List, iou_threshold=0.5):
    matched_pred = set()
    matched_gt = set()

    iou_scores = []

    for i, gt in enumerate(ground_truth):
        best_iou = 0
        best_j = -1
        for j, pred in enumerate(predicted):
            if j in matched_pred:
                continue
            score = iou(gt, pred)
            if score > best_iou:
                best_iou = score
                best_j = j
        if best_iou >= iou_threshold and best_j != -1:
            matched_pred.add(best_j)
            matched_gt.add(i)
            iou_scores.append(best_iou)

    precision = len(matched_pred) / len(predicted) if predicted else 0
    recall = len(matched_gt) / len(ground_truth) if ground_truth else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

    return {
        "matched_pairs": len(iou_scores),
        "average_iou": np.mean(iou_scores) if iou_scores else 0,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def global_jaccard(a, b) -> float:
    """Jaccard similarity over total union."""
    all_intervals = a + b
    if not all_intervals:
        return 1.0

    min_start = min(i[0] for i in all_intervals)
    max_end = max(i[1] for i in all_intervals)

    total_union = 0
    total_intersection = 0

    for t in range(int(min_start), int(max_end)):
        in_a = any(start <= t < end for (start, end) in a)
        in_b = any(start <= t < end for (start, end) in b)
        if in_a or in_b:
            total_union += 1
        if in_a and in_b:
            total_intersection += 1

    return total_intersection / total_union if total_union > 0 else 1.0