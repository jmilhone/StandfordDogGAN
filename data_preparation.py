import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def parse_dog_xml(filename, image_margin=5):
    tree = ET.parse(filename)
    root = tree.getroot()

    image_filename = root.find('filename').text + ".jpg"

    print(image_filename)
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    objects = root.findall('object')
    objects_data = list()

    for idx, object in enumerate(objects):
        object_data = dict()
        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        xmax = int(bbox.find('xmax').text)
        ymin = int(bbox.find('ymin').text)
        ymax = int(bbox.find('ymax').text)

        xmin -= image_margin
        xmax += image_margin
        ymin -= image_margin
        ymax += image_margin

        if xmin < 0:
            xmin = 0

        if ymin < 0:
            ymin = 0

        # slices are inclusive on left and exclusive on the right
        if xmax >= width-1:
            xmax = width

        if ymax >= height-1:
            ymax = height

        w = min(xmax - xmin, ymax-ymin)

        object_data['filename'] = image_filename
        object_data['width'] = width
        object_data['height'] = height
        object_data['xmin'] = xmin
        object_data['ymin'] = ymin
        object_data['xmax'] = xmax
        object_data['ymax'] = ymax
        objects_data.append(object_data)
    return objects_data

if __name__ == "__main__":
    filename = os.path.join(
        "Annotation",
        #"n02085620-Chihuahua",
        "n02111500-Great_Pyrenees",
        #"n02085620_1816"
        "n02111500_3835",
    )
    info = parse_dog_xml(filename)
    print(info[0])
    info = info[0]

    image_filename = os.path.join(
        "Images",
        #"n02085620-Chihuahua",
        "n02111500-Great_Pyrenees",
        info['filename']
    )

    img = cv2.imread(image_filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    w = min(
        info['xmax']-info['xmin'],
        info['ymax']-info['ymin'],
    )

    mid = ((info['xmax']+info['xmin'])/2.0, (info['ymax']+info['ymin'])/2.0)
    left_corner = (mid[0]-w/2.0, mid[1]-w/2.0)
    print(w, mid)
    fig, ax = plt.subplots()
    ax.imshow(img)
    rect = patches.Rectangle((info['xmin'], info['ymin']),
                             info['xmax']-info['xmin'],
                             info['ymax']-info['ymin'], linewidth=1,
                             edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle(left_corner, w, w,
                              linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect)
    ax.add_patch(rect2)
    plt.show()


