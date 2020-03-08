# instance-dominant-colors
Extract dominant colors from an instance segment.

Instance segmentstion method: [Mask R-CNN](https://github.com/matterport/Mask_RCNN) with its provided trained COCO model.

Extract dominant colors method: [k-means clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

# Example
pizza
<p align="center">
    <img src="sample_img/pizza.jpg" width="400">
    <img src="sample_img/pizza_inst.jpg" width="400">
    <br>Photo by mahyar motebassem on Unsplash</br>
    <img src="sample_img/pizza_out.jpg" width="300">
</p>

traffic light
<p align="center">
    <img src="sample_img/trafficlight.jpg" width="400">
    <img src="sample_img/trafficlight_inst.jpg" width="400">
    <br>Photo by Aleksandr Kotlyar on Unsplash</br>
    <img src="sample_img/trafficlight_out.jpg" width="300">
</p>

dog
<p align="center">
    <img src="sample_img/dog.jpg" width="400">
    <img src="sample_img/dog_inst.jpg" width="400">
    <img src="sample_img/dog_out.jpg" width="300">
</p>

# Installation
Clone this repository

```
$ cd instance-dominant-colors
$ pip3 install -r requirements.txt
```

# Usage
```python
from instance2color import Instance2Color

inst = Instance2Color(image_file='sample_img/pizza.jpg', class_name='pizza', num_of_color=5)

print(inst.dominant_colors()) # RGBs and percentages
# [([173, 76, 67], 29.44), ([222, 139, 100], 26.64), ([98, 46, 38], 19.68), ([96, 181, 108], 12.9), ([227, 205, 188], 11.33)]
inst.visualize_pie(output_file='sample_img/pizza_out.jpg')
# output pie chart
inst.display_instance()
# visualize instance segmentation
```





