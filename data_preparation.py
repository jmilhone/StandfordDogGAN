import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import concurrent.futures
from tqdm import tqdm
import random
import time


def parse_dog_xml(filename, folder, image_margin=5, only_single_dog=True):
    tree = ET.parse(filename)
    root = tree.getroot()

    image_filename = filename + ".jpg"

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    objects = root.findall('object')
    objects_data = list()

    for idx, obj in enumerate(objects):
        object_data = dict()
        bbox = obj.find('bndbox')
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
        object_data['breed'] = obj.find('name').text
        object_data['folder'] = folder
        objects_data.append(object_data)

    if only_single_dog and len(objects_data) > 1:
        return None

    return objects_data

def parse_all_annotations(folder):

    objects_data = list()
    futures = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for directory in os.listdir(folder):
            breed_directory = os.path.join(folder, directory)
            futures += [executor.submit(parse_dog_xml, os.path.join(folder, directory, image_xml), directory) for image_xml in os.listdir(breed_directory)]
                # objects_data.extend(parse_dog_xml(fname))
        n_futures = len(futures)
        for future in tqdm(concurrent.futures.as_completed(futures), total=n_futures):
            result = future.result()
            if result is not None:
                objects_data.extend(future.result())

    print(len(objects_data))
    return objects_data

def prep_single_image(image_info, final_image_size=64, plotit=False):
    filename = image_info['filename']
    breed_folder = image_info['folder']
    filename = filename.replace("Annotation", "Images")

    # filepath = os.path.join('Images', breed_folder, filename)
    # filepath = os.path.join(breed_folder, filename)

    img = cv2.imread(filename)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ymin = image_info['ymin']
    ymax = image_info['ymax']
    xmin = image_info['xmin']
    xmax = image_info['xmax']

    width = image_info['width']
    height = image_info['height']

    w = max(xmax-xmin,ymax-ymin)

    mid = ((xmax+xmin)/2.0, (ymax+ymin)/2.0)
    # I want a square bounding box. It would be nice to increase the bounding box instead of shrinking it
    # ToDo: think of a better bounding box adjustment that is still square.
    if mid[0]+w/2.0 > width or mid[0]-w/2.0 < 0:
        #print("w is too big in the x direction")
        #print(f"old w: {w}")
        w = min(width-mid[0], mid[0])*2
        #print(f"new w: {w}")
    elif mid[1]+w/2.0 > height or mid[1]-w/2.0 < 0:
        #print('w is too big for the y direction')
        #print(f"old w: {w}")
        w = min(height-mid[1], mid[1])*2
        #print(f"new w: {w}")

    left_corner = (mid[0]-w/2.0, mid[1]-w/2.0)
    #print(w, mid)
    if plotit:
        fig, ax = plt.subplots()
        ax.imshow(img)
        rect = patches.Rectangle((xmin, ymin),
                                 xmax-xmin,
                                 ymax-ymin, linewidth=1,
                                 edgecolor='r', facecolor='none')
        rect2 = patches.Rectangle(left_corner, w, w,
                                  linewidth=1, edgecolor='c', facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)
        plt.show(block=False)

    img_cropped = img[ymin:ymin+int(w), xmin:xmin+int(w), :]
    if int(w) > final_image_size:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    img_cropped = cv2.resize(img_cropped, (final_image_size, final_image_size), interpolation=interpolation)

    if plotit:
        fig, ax = plt.subplots()
        ax.imshow(img_cropped)
        plt.show()

    return img_cropped

def prep_all_images(images_info, final_image_size=64):
    n_images = len(images_info)
    all_images = np.zeros((n_images, final_image_size, final_image_size, 3), dtype=int)

    # for i, info in enumerate(images_info):
    #     all_images[i, :, :, :] = prep_single_image(info, plotit=False)
    # print(time.time()-t0, 'seconds')

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        t0 = time.time()
        future_map = dict()
        for i, info in enumerate(images_info):
            future = executor.submit(prep_single_image, info)
            future_map[future] = i

        for future in concurrent.futures.as_completed(future_map):
            index = future_map[future]
            all_images[index, :, :, :] = future.result()
        t1 = time.time()
        print(t1-t0, 'seconds')
    return all_images

if __name__ == "__main__":
    dog_info = parse_all_annotations("Annotation")
    all_dog_images = prep_all_images(dog_info)

    # Grab 16 random dog images to show
    subset = random.choices(range(all_dog_images.shape[0]), k=16)
    fig = plt.subplots(4, 4)
    for i in range(16):
        index = subset[i]
        plt.subplot(4, 4, i+1)
        plt.imshow(all_dog_images[index, :, :, :])
        plt.axis('off')
    plt.show()

