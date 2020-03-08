from inst_segment import segment_color_list
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from mrcnn.visualize import apply_mask
from PIL import Image


def cluster_percents(labels):
    total = len(labels)
    percents = []
    for i in set(labels):
        percent = (np.count_nonzero(labels == i) / total) * 100
        percents.append(round(percent, 2))
    return percents


class Instance2Color:
    def __init__(self, image_file, class_name, num_of_color=5):
        self.rgbs, self.inst_info = segment_color_list(image_file, class_name)
        self.cluster = KMeans(n_clusters=num_of_color).fit(self.rgbs)

    def visualize_instance(self, output_file):
        image, _, mask, _, _ = self.inst_info
        mask = mask[:, :, 0]
        image_masked = apply_mask(image, mask, [1, 0, 0], alpha=0.6)
        im = Image.fromarray(image_masked)
        im.save(output_file)

    def dominant_colors(self):
        colors = np.round(self.cluster.cluster_centers_).astype(np.uint8).tolist()
        percents = cluster_percents(self.cluster.labels_)
        tup = zip(colors, percents)
        sorted_tup = sorted(tup, key=lambda n: n[1], reverse=True)
        return sorted_tup

    def visualize_pie(self, output_file):
        sorted_tup = self.dominant_colors()
        colors = [c for c, p in sorted_tup]
        colors = [list(map(lambda n: n / 255, c)) for c in colors]
        percents = [p for c, p in sorted_tup]
        plt.pie(percents, colors=colors, counterclock=False, startangle=90)
        plt.savefig(output_file, bbox_inches='tight')
