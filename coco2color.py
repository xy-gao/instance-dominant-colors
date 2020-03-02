from inst_segment import segment_color_list
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


def cluster_percents(labels):
    total = len(labels)
    percents = []
    for i in set(labels):
        percent = (np.count_nonzero(labels == i) / total) * 100
        percents.append(round(percent, 2))
    return percents


class Coco2Color:
    def __init__(self, image, class_name, num_of_color=5):
        self.image = image
        self.class_name = class_name
        self.num_of_color = num_of_color
        self.rgbs = segment_color_list(self.image, self.class_name)
        self.cluster = KMeans(n_clusters=num_of_color).fit(self.rgbs)

    def dominant_colors(self):
        colors = self.cluster.cluster_centers_.astype(int).tolist()
        percents = cluster_percents(self.cluster.labels_)
        tup = zip(colors, percents)
        sorted_tup = sorted(tup, key=lambda n: n[1], reverse=True)
        return sorted_tup

    def visualize(self, output_path):
        sorted_tup = self.dominant_colors()
        colors = [c for c, p in sorted_tup]
        colors = [list(map(lambda n: n / 255, c)) for c in colors]
        percents = [p for c, p in sorted_tup]
        plt.pie(percents, colors=colors, counterclock=False, startangle=90)
        plt.savefig(output_path)
