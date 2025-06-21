from sklearn.cluster import DBSCAN
import numpy as np

def cluster_time_points(points):
    X=np.array(points)
    X= X.reshape(-1, 1)
    db = DBSCAN(eps=10, min_samples=2).fit(X)

    labels = db.labels_

    from collections import defaultdict
    clusters = defaultdict(list)
    for point, label in zip(X, labels):
        if label != -1:
            clusters[label].append(point)

    # Output clusters
    return clusters

def find_intervals_with_clusterization(detected_change_points):

    list_of_lists = []

    groups = cluster_time_points(detected_change_points)
    for key,value in groups.items():
        flat_list = [x[0] for x in value]
        # print(flat_list)
        list_of_lists.append(flat_list)

    intervals = []
    for group in list_of_lists:
        if len(group) > 2:
            intervals.append((group[0], group[-1]))
    return intervals