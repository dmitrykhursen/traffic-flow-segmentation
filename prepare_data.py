import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from fastseg.image.colorize import colorize, blend

PATH_TO_DATA = "./data/highway_dataset"
path_to_rgb = f"{PATH_TO_DATA}/rgb/"
path_to_vis = f"{PATH_TO_DATA}/vis/"
path_to_seg = f"{PATH_TO_DATA}/seg/"

name_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged', 'unsegmented']


def access_to_seg_data(filename: str) -> None:
    seg_file = path_to_seg + filename + ".npy"
    seg_data = np.load(seg_file)


    clas = name_list[seg_data[250][250]] # semantic class of the value
    print(clas)
    assert seg_data.shape == (512, 910), f"seg_data shape ISN'T (512, 910)"
    print(seg_data.shape) # shape is (512, 910)
    print(seg_data[0, :])
    print(seg_data)

    subset_arr = seg_data[:, :513]
    print(subset_arr.shape)
    print(subset_arr)

    rgb_img = path_to_rgb + filename + ".png"
    img = Image.open(rgb_img).resize((910, 512))
    # img.show()
    
    all_classes_on_the_img = np.unique(seg_data)

    str_class_names = []
    for c in all_classes_on_the_img:
        str_class_names.append(name_list[c])
    
    print(all_classes_on_the_img)
    print(str_class_names)
    
    # show_img(img, seg_data)

def process_npy_files(folder_path):
    # Check if the folder path exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return
    
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)
    
    # Filter .npy files
    npy_files = [f for f in file_list if f.endswith('.npy')]
    
    if not npy_files:
        print(f"No .npy files found in '{folder_path}'.")
        return
    
    classes_on_all_images = []
    unique_values = set()    
    for file_name in npy_files:
        file_path = os.path.join(folder_path, file_name)
        
        # Load the .npy file
        try:
            seg_data = np.load(file_path)
            classes_on_all_images.append(np.unique(seg_data))
            # unique_values.update(seg_data)
            num = 11
            if num in np.unique(seg_data):
                print(f"class {name_list[num]} is in image:   {file_name}")
                # break
            
        except Exception as e:
            print(f"Error processing '{file_name}': {e}")
            break
            # continue

    # print(classes_on_all_images)
    # classes_on_all_images = np.array(classes_on_all_images) 
    global_classes_on_all_images = np.unique(np.concatenate(classes_on_all_images))
    # global_classes_on_all_images = sorted(unique_values)

    str_classes = []
    for c in global_classes_on_all_images:
        str_classes.append(name_list[c])
    
    # print(classes_on_all_images)
    print(global_classes_on_all_images)
    print(len(global_classes_on_all_images))
    print()
    print(str_classes)
    

def show_img(img: Image, seg_np: np.array) -> None:
    # Calculation of final segmentation prediction from class probabilities along dimension 1
    # detach.cpu.numpy transfer tensor from torch to computational graph-detached, to cpu memory and to numpy array instead of tensor
    # seg_np = output.argmax(dim=1)[0].detach().cpu().numpy()

    # Function from fastseg to visualize images and output segmentation
    seg_img = colorize(seg_np) # <---- input is numpy, output is PIL.Image
    # seg_img.show()
    # return
    blended_img = blend(img, seg_img, factor=0.75) # <---- input is PIL.Image in both arguments

    # Concatenate images for simultaneous view
    new_array = np.concatenate((np.asarray(blended_img), np.asarray(seg_img)), axis=1)

    # Show image from PIL.Image class
    combination = Image.fromarray(new_array)
    combination.show()



if __name__ == "__main__":
    print("Hello world!")

    # filename = "102600107"
    # access_to_seg_data(filename)

    # process/iterate through folder (all files)
    process_npy_files(path_to_seg)

# in higway dataset all the unique classes are:
# ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'stop sign', 'bridge', 'house', 'light', 'railroad', 'road', 'wall-stone', 'water-other', 'tree-merged', 'fence-merged', 'sky-other-merged', 'pavement-merged', 'grass-merged', 'dirt-merged', 'building-other-merged', 'wall-other-merged', 'unsegmented']
