from typing import List, Tuple

def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged

def interval_length(intervals: List[Tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)

def intersect_intervals(a: List[Tuple[int, int]], b: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    i = j = 0
    intersection = []
    while i < len(a) and j < len(b):
        a_start, a_end = a[i]
        b_start, b_end = b[j]
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        if start < end:
            intersection.append((start, end))
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return intersection

def dice_coefficient(a: List[Tuple[int, int]], b: List[Tuple[int, int]]) -> float:
    a_merged = merge_intervals(a)
    b_merged = merge_intervals(b)
    len_a = interval_length(a_merged)
    len_b = interval_length(b_merged)
    inter = intersect_intervals(a_merged, b_merged)
    len_inter = interval_length(inter)
    if len_a + len_b == 0:
        return 1.0  # define as 1 if both are empty
    return 2 * len_inter / (len_a + len_b)
