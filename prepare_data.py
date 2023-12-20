import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
from copy import deepcopy

from fastseg.image.colorize import colorize, blend

PATH_TO_DATA = "./data/highway_dataset"
path_to_rgb = f"{PATH_TO_DATA}/rgb/"
path_to_vis = f"{PATH_TO_DATA}/vis/"
# path_to_seg = f"{PATH_TO_DATA}/seg/"
path_to_seg = f"/home/dkhursen/Documents/B3B33UROB/HW_03_segmentation/traffic-flow-segmentation/data/filtered_dataset/seg/"


name_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged', 'unsegmented']


def access_to_seg_data(filename: str) -> None:
    seg_file = path_to_seg + filename + ".npy"
    seg_data = np.load(seg_file)

    assert seg_data.shape == (512, 910), f"seg_data shape ISN'T (512, 910)"
    print(seg_data.shape) # shape is (512, 910)
    print(seg_data)

    rgb_img = path_to_rgb + filename + ".png"
    img = Image.open(rgb_img).resize((910, 512))
    # img.show()
    
    all_classes_on_the_img = np.unique(seg_data)

    str_class_names = []
    for c in all_classes_on_the_img:
        str_class_names.append(name_list[c])
    
    print(all_classes_on_the_img)
    print(str_class_names)
    visualize_img_and_blended_seg(img, seg_data)

def process_npy_files(folder_path: str) -> None:
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
            # num = 11
            # if num in np.unique(seg_data):
            #     print(f"class {name_list[num]} is in image:   {file_name}")
            #     # break
            
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
    

def visualize_img_and_blended_seg(img: Image, seg_np: np.array) -> None:
    # Calculation of final segmentation prediction from class probabilities along dimension 1
    # detach.cpu.numpy transfer tensor from torch to computational graph-detached, to cpu memory and to numpy array instead of tensor
    # seg_np = output.argmax(dim=1)[0].detach().cpu().numpy()

    # just to changes the colors for visualization
    # seg_np[seg_np == 2] = 6 ### delete this
    # seg_np[seg_np == 100] = 2 ### delete this
    # seg_np[seg_np == 133] = 5 ### delete this
    # seg_np[seg_np == 7] = 1 ### delete this


    # Function from fastseg to visualize images and output segmentation
    seg_img = colorize(seg_np) # <---- input is numpy, output is PIL.Image
    # seg_img.show()
    blended_img = blend(img, seg_img, factor=0.75) # <---- input is PIL.Image in both arguments
    # blended_img.show()

    # Concatenate images for simultaneous view
    new_array = np.concatenate((np.asarray(blended_img), np.asarray(seg_img)), axis=1)

    # Show image from PIL.Image class
    combination = Image.fromarray(new_array)
    combination.show()

def export_data_with_classes_to_use(folder_path: str, classes_to_use: list, new_folder_path: str) -> None:
    """ Creates a new folder with seg data witn only classes to use/keep """
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)
    
    # Filter .npy files
    npy_files = [f for f in file_list if f.endswith('.npy')]
    
    if not npy_files:
        print(f"No .npy files found in '{folder_path}'.")
        return
    
    n_files = len(npy_files)
    classes_distribution = dict()
    new_unsegmented_id = keep_classes.index("unsegmented")
    
    for file_name in npy_files:
        file_path = os.path.join(folder_path, file_name)
        
        seg_data = np.load(file_path)
        # a = Image.fromarray(seg_data)
        # print(file_name)
        # a.show()

        new_seg_data = deepcopy(seg_data)

        classes_in_the_image = np.unique(seg_data)
        for c in classes_in_the_image:

            if not name_list[c] in classes_to_use:
                # mark this class as unsegmented
                new_seg_data[new_seg_data == c] = new_unsegmented_id
            else:
                new_index = keep_classes.index(name_list[c])
                new_seg_data[new_seg_data == c] = new_index


        np.save(new_folder_path+file_name, new_seg_data)
        # Image.fromarray(new_seg_data).show()
        # time.sleep(5)

def show_class_distribution(folder_path: str) -> None:
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)
    
    # Filter .npy files
    npy_files = [f for f in file_list if f.endswith('.npy')]
    
    if not npy_files:
        print(f"No .npy files found in '{folder_path}'.")
        return
    
    n_files = len(npy_files)
    classes_distribution = dict()
    unique_values = set()    
    for file_name in npy_files:
        file_path = os.path.join(folder_path, file_name)
        
        seg_data = np.load(file_path)
        classes_in_the_image = np.unique(seg_data)
        
        # add found classes to dict class distribution
        for c in classes_in_the_image:
            if name_list[c] in classes_distribution:
                classes_distribution[name_list[c]] += 1
            else:
                classes_distribution[name_list[c]] = 1
    
    print(classes_distribution)
    print(f"amount of imgs: {n_files}")
    class_distribution_in_percentage = {str_class: (classes_distribution[str_class]/n_files)*100 for str_class in classes_distribution}
    print(class_distribution_in_percentage)

    # class_names = list(classes_distribution.keys())
    # class_amounts = list(classes_distribution.values())

    class_names = list(class_distribution_in_percentage.keys())
    class_amounts = list(class_distribution_in_percentage.values())

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_amounts)
    plt.xlabel('Class')
    plt.ylabel('Percentage of images containig the class')
    plt.title('Classes Distribution')
    plt.xticks(rotation=90)  # Rotating x-axis labels for better readability
    plt.tight_layout()

    plt.show()



if __name__ == "__main__":
    filename = "102600107"
    access_to_seg_data(filename)

    # process/iterate through folder (all files)
    # process_npy_files(path_to_seg)

    # make and show classes distribution
    # show_class_distribution(path_to_seg)
    # show_class_distribution("/home/dkhursen/Documents/B3B33UROB/HW_03_segmentation/segmentation_for_traffic_flow_analysis/datasets/intercity_crossroad_from_drone_dataset/seg/")

    # filter dataset to my usecase
    # keep_classes = ["bus", "truck", "road", "motorcycle", "person", 'unsegmented', "car"]
    # new_folder_path = "/home/dkhursen/Documents/B3B33UROB/HW_03_segmentation/traffic-flow-segmentation/data/filtered_dataset/seg/"
    # export_data_with_classes_to_use(path_to_seg, keep_classes, new_folder_path)

# in higway dataset all the unique classes are:
# ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'stop sign', 'bridge', 'house', 'light', 'railroad', 'road', 'wall-stone', 'water-other', 'tree-merged', 'fence-merged', 'sky-other-merged', 'pavement-merged', 'grass-merged', 'dirt-merged', 'building-other-merged', 'wall-other-merged', 'unsegmented']
