import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import scipy
import time
import pickle
import io
import os
import cv2

from fastseg import MobileV3Small
from fastseg.image.colorize import colorize, blend
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import HigwWayDataset
from tqdm import tqdm
# from sklearn.metrics import confusion_matrix


# filtered classes for my case
classes = ["bus", "truck", "road", "motorcycle", "person", 'unsegmented', "car"]

def load_dataset(path_to_dataset_folder: str, train_size: float, batch_size: int) -> (DataLoader, DataLoader):
    """ Loads the dataset: images(features) are in 'rgb' folder and labels are in 'seg' folder. Train size is a float (0.0, 1.0)
        Returns train and test dataloaders"""
    
    transform = transforms.Compose([
    transforms.Resize((512, 910)),  # Resize images
    transforms.ToTensor()  # Convert images to PyTorch tensors
    ])
    dataset = HigwWayDataset(path_to_dataset_folder, transform)
    # Split the dataset
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size  # Remaining for test
    train_set, test_set = random_split(dataset, [train_size, test_size])

    # Create DataLoader for train and test sets
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_model(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim, criterion: torch.nn, epochs: int = 100, save_model=False):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Processing epoch {epoch+1}"):
            # Zero the parameter gradients
            optimizer.zero_grad()

            labels = labels.to(torch.long) #

            # Forward pass
            outputs = model(inputs)

            # print(np.unique(labels.numpy()))
            # print(np.unique(outputs.detach().numpy()))
            loss = criterion(outputs, labels)
            # print(loss)
            # print(loss.size())
            # print(loss.mean())

            # Backward pass and optimize
            loss.mean().backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.mean().item()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}")

    # # Visualization of model's output at every iterations
    # seg_np = output.argmax(dim=1)[0].detach().cpu().numpy()
    # seg_img = colorize(seg_np)
    # seg_img.save(f"/home/dkhursen/Documents/B3B33UROB/HW_03_segmentation/traffic-flow-segmentation/overfitting/{e:03d}.png")

    # Saving weights
    if save_model:
        torch.save(model.state_dict(), 'weights/model.pth')

def get_test_metrics(labels, predictions):
    # labels = labels.cpu().numpy()
    # predictions = predictions.detach().cpu().numpy()

    labels = labels.cpu().numpy()
    predictions = predictions.detach().cpu().numpy()

    predictions = np.argmax(predictions, 0)  # first dimension are probabilities/scores
    # print(predictions.shape)

    # tmp_cm = scipy.sparse.coo_matrix(
    #     (np.ones(np.prod(labels.shape), 'u8'), (labels.flatten(), predictions.flatten())), shape=(7, 7)
    # ).toarray()  # Fastest possible way to create confusion matrix. Speed is the necessity here, even then it takes quite too much
    # tmp_cm = confusion_matrix(labels.flatten(), predictions.flatten())

    # true_masks_flat = labels.flatten()
    # predicted_masks_flat = predictions.flatten()
    # # Initialize the confusion matrix with zeros
    # num_classes = 7
    # tmp_cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    # # Fill the matrix by comparing true and predicted masks
    # for i in range(len(true_masks_flat)):
    #     true_label = true_masks_flat[i]
    #     predicted_label = predicted_masks_flat[i]
    #     tmp_cm[true_label][predicted_label] += 1
    
    combined_labels = labels * num_classes + predictions
    # Create the confusion matrix using bincount and reshaping
    tmp_cm = np.bincount(
        combined_labels.ravel(), minlength=num_classes**2
    ).reshape(num_classes, num_classes)

    tps = np.diag(tmp_cm)
    fps = tmp_cm.sum(0) - tps
    fns = tmp_cm.sum(1) - tps

    tps = np.nan_to_num(tps, nan=0.0)
    fps = np.nan_to_num(fps, nan=0.0)
    fns = np.nan_to_num(fns, nan=0.0)

    with np.errstate(all='ignore'):
        precisions = tps / (tps + fps)
        recalls = tps / (tps + fns)
        ious = tps / (tps + fps + fns)

    return precisions, recalls, ious

def test_model_segmentation(model: nn.Module, test_loader: DataLoader, criterion: torch.nn, load_model=False, weights_path="") -> float:
    if load_model:
        print("Loading model weights ", weights_path)
        # Load the weights into the model
        model.load_state_dict(torch.load(weights_path))

    model.eval()  # Set the model to evaluation mode

    iou_sum = 0.0
    num_samples = 0
    tps, fps, fns = 0, 0, 0

    precisions = np.zeros((7,))
    recalls = np.zeros((7,))
    ious = np.zeros((7,))


    with torch.no_grad():
        for inputs, true_masks in tqdm(test_loader):
            output = model(inputs)
            # pred_masks = output.argmax(dim=1)[0].detach().cpu().numpy()
            pred_masks = output.argmax(dim=1)
            
            num_samples += 1
            # print("!!!!!!")
            tps, fps, fns = update(true_masks, output)
            with np.errstate(all='ignore'):
                precisions += tps / (tps + fps)
                recalls += tps / (tps + fns)
                ious += tps / (tps + fps + fns)

    print("---")
    print("Mprecisions:", precisions/num_samples)
    print("Mrecalls:", recalls/num_samples)
    print("Mious:", ious/num_samples)
    mean_iou = iou_sum / num_samples
    # print(f"Mean IoU: {mean_iou:.4f}")
    return mean_iou

def train_model_with_test(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim, criterion: torch.nn, epochs: int, model_info: str, device_str: str, save_stats: bool):
    # model.train()
    # statistics to save
    train_losses = []
    test_losses = []
    test_precision = []
    test_recall = []
    test_iou = []
    test_mean_iou = []
    test_mean_iou2 = []

    if device_str == "cuda":
        device_str += ":2"

    print(f"device_str: {device_str}")
    device = torch.device(device_str)
    print(f"Device: {device}")
    model = model.to(device)
    start_training_time = time.time()
    for epoch in range(epochs):
        train_loss = 0.0
        # train loop
        for inputs, labels in tqdm(train_loader, desc=f"Training epoch {epoch+1}"):
            # Zero the parameter gradients
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.long)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Backward pass and optimize
            loss.mean().backward()
            optimizer.step()

            train_loss += loss.mean().item()
        
        train_losses.append(train_loss / len(train_loader))

        # test loop
        test_loss = 0.0
        epoch_mean_iou = []
        epoch_mean_iou2 = []
        for test_inputs, test_labels in tqdm(test_loader, desc=f"Evaluating epoch {epoch+1}"):
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device, dtype=torch.long)

            output = model(test_inputs)
            t_loss = criterion(output, test_labels)
            test_loss += t_loss.mean().item()

            # print(output.shape)
            # p = []
            # r = []
            m_ious = []
            m_ious2 = []
            for label, predict in zip(test_labels, output):
                # print(predict.shape)
                # print(label.shape)

                precision, recall, iou = get_test_metrics(label, predict)
                # precision = np.nan_to_num(precision, nan=0.0)
                # recall = np.nan_to_num(recall, nan=0.0)
                iou = np.nan_to_num(iou, nan=0.0)

                m_iou2 = np.ma.average(iou, weights=[0,1,1,0,0,0,1]) # care only about cars, trucks, road
                m_iou = np.ma.average(iou, weights=[0,0,1,0,0,0,1]) # care only about cars, road
                m_ious.append(m_iou)
                m_ious2.append(m_iou2)
            
            mean_iou = np.average(m_ious)
            mean_iou2 = np.average(m_ious2)
            # print(mean_iou)
            # break

            # mean_iou2 = np.ma.average(iou, weights=[1,1,1,1,1,0,1]) #
            # # mean_iou = np.ma.average(iou, weights=[0,1,1,0,0,0,1]) # care only about cars, trucks, road
            # mean_iou3 = np.ma.average(iou, weights=[0,0,1,0,0,0,1]) # care only about cars, road
            # mean_iou4 = np.average(iou) #
            # break


            # test_precision.append(precision)
            # test_recall.append(recall)
            # test_iou.append(iou)
            epoch_mean_iou.append(mean_iou)
            epoch_mean_iou2.append(mean_iou2)

        test_mean_iou.append(np.average(epoch_mean_iou))
        test_mean_iou2.append(np.average(epoch_mean_iou2))
        test_losses.append(test_loss / len(test_loader))
        print(f"Epoch [{epoch + 1}/{epochs}], Train loss: {train_loss / len(train_loader)} , Test loss: {test_loss / len(test_loader)}, Mean IoU: {np.average(epoch_mean_iou)}, Mean IoU2: {np.average(epoch_mean_iou2)}")
    
    end_training_time = time.time()
    training_time = end_training_time - start_training_time

    # save statistics as dict
    stats = {}
    stats["model info"] = model_info
    stats["device"] = device_str
    stats["weights"] = model.state_dict()
    stats["train loss"] = np.round(train_losses, 3)
    stats["test loss"] = np.round(test_losses, 3)
    stats["train time"] = np.round(training_time, 2)

    # stats["test precision"] = test_precision
    # stats["test recall"] = test_recall
    # stats["test IoU"] = test_iou
    stats["test mean IoU"] = np.round(test_mean_iou, 3)
    stats["test mean IoU 2"] = np.round(test_mean_iou2, 3)

    if save_stats:
        print(f"Saving stast in {model_info}.pkl")
        with open(f'stats/{model_info}.pkl', 'wb') as file:
            pickle.dump(stats, file)

        print(f"Saving model in {model_info}.pth")
        torch.save(model.state_dict(), f'weights/{model_info}.pth')

def print_stats(model_info: str):
    with open(f'stats/{model_info}.pkl', 'rb') as file:
        loaded_dict = pickle.load(file)
    
    loaded_dict["weights"] = loaded_dict["weights"]["last.bias"]
    print(loaded_dict)

def show_statistics(model_info: str):
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else:
                return super().find_class(module, name)
        
    with open(f'gpu/stats/{model_info}.pkl', 'rb') as file:
    #     # data = pickle.load(file, map_location='cpu')
    #     data = torch.load(file, map_location=torch.device('cpu'))
        data = CPU_Unpickler(file).load()

    print("Model: ", data["model info"])
    print("Device: ", data["device"])
    print("Training time: ", data["train time"])

    # stats["train time"] = np.round(training_time, 2)
        
    epochs = range(1, len(data["train loss"]) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, data["train loss"], 'b', label='Train Loss')
    plt.plot(epochs, data["test loss"], 'r', label='Test Loss')
    plt.title(f'{data["model info"]}: Train and Test Losses Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, data["test mean IoU"], 'orange', label='Mean IoU 1')
    plt.plot(epochs, data["test mean IoU 2"], 'purple', label='Mean IoU 2')
    plt.title(f'{data["model info"]}: Mean IoU 1 and Mean IoU 2 Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean IoU')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_model_segmentation(model: nn.Module, test_loader: DataLoader, load_model=False, weights_path="") -> float:
    if load_model:
        print("Loading model weights ", weights_path)
        # Load the weights into the model
        model.load_state_dict(torch.load(weights_path))

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for inputs, true_masks in tqdm(test_loader):
            output = model(inputs)
            pred_masks = output.argmax(dim=1)[0].detach().cpu().numpy()

def access_to_seg_data(filename: str, model) -> None:
    path_to_rgb = "./data/filtered_dataset/rgb/"
    rgb_img = path_to_rgb + filename + ".png"
    img = Image.open(rgb_img)
    # img.show()
    transform = transforms.Compose([
    transforms.Resize((512, 910)),  # Resize images
    transforms.ToTensor()
    ])
    input = transform(img).unsqueeze(0)

    print("here")
    print(input.shape)
    with torch.no_grad():
        output = model(input)
        seg_data = output.argmax(dim=1)[0].detach().cpu().numpy()
    
    print(seg_data.shape)
    visualize_img_and_blended_seg(img.resize((910, 512)), seg_data)


def visualize_img_and_blended_seg(img: Image, seg_np: np.array) -> None:
    # Calculation of final segmentation prediction from class probabilities along dimension 1
    # detach.cpu.numpy transfer tensor from torch to computational graph-detached, to cpu memory and to numpy array instead of tensor
    # seg_np = output.argmax(dim=1)[0].detach().cpu().numpy()

    # Function from fastseg to visualize images and output segmentation
    seg_img = colorize(seg_np) # <---- input is numpy, output is PIL.Image
    # seg_img.show()
    blended_img = blend(img, seg_img, factor=0.55) # <---- input is PIL.Image in both arguments
    # blended_img.show()

    # Concatenate images for simultaneous view
    new_array = np.concatenate((np.asarray(blended_img), np.asarray(seg_img)), axis=1)

    # Show image from PIL.Image class
    combination = Image.fromarray(new_array)
    combination.show()

def make_video_output():
    width = 910
    height = 512
    transform = transforms.Compose([
        transforms.Resize((height, width)),  # Replace with desired height and width
        transforms.ToTensor()
        # Add other necessary transformations
    ])
    model_path = "./gpu/weights/adam_model_100-epochs.pth"
    num_classes = 7
    model = MobileV3Small(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    input_folder = './data/filtered_dataset/test_imgs_for_video/rgb'
    output_folder = './data/filtered_dataset/test_imgs_for_video/combined_output'

    # Create a list to store segmented image paths
    segmented_image_paths = []

    # Loop through the images in the input folder
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".png"):
            # Load the image
            # image = Image.open(os.path.join(input_folder, filename))
            # # # # Preprocess the image
            # image_tensor = transform(image).unsqueeze(0)
            # image = image.resize((width, height))
            
            # # # # Apply your segmentation model to get segmented masks/predictions
            # output = model(image_tensor)
            # segmented_mask = output.argmax(dim=1)[0].detach().cpu().numpy()
            
            # # # # Save the segmented mask
            output_path = os.path.join(output_folder, filename)
            # # # # cv2.imwrite(output_path, segmented_mask)
            # # # colorize(segmented_mask).save(output_path)
            # # blend(image, colorize(segmented_mask), factor=0.55).save(output_path)

            # seg_img = colorize(segmented_mask) 
            # blended_img = blend(image, seg_img, factor=0.55)
            # # Concatenate images for simultaneous view
            # new_array = np.concatenate((np.asarray(blended_img), np.asarray(seg_img)), axis=1)
            # # Show image from PIL.Image class
            # combination = Image.fromarray(new_array)
            # combination.save(output_path)
            
            segmented_image_paths.append(output_path)

    # Create a video from the segmented images
    video_output_path = './data/filtered_dataset/test_imgs_for_video/video.avi'
    fps = 30  # Adjust frames per second

    # Load the first image to get the size
    first_image = Image.open(segmented_image_paths[0])
    # height, width, layers = first_image.shape

    # video = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

    video = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width, height))

    print(len(segmented_image_paths))
    for image_path in tqdm(segmented_image_paths):
        # video.write(cv2.imread(image_path))
        pil_img = Image.open(image_path)
        # Resize the image to match the video frame size (if needed)
        pil_img = pil_img.resize((width, height))
        # Convert PIL image to OpenCV compatible format (numpy array)
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        video.write(cv_img)

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    make_video_output()

# if __name__ == "__main__":
#     filename = "102600166"
#     model_path = "./gpu/weights/adam_model_100-epochs.pth"
#     num_classes = 7
#     model = MobileV3Small(num_classes=num_classes)
#     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

#     access_to_seg_data(filename, model)

# if __name__ == "__main__":
#     # epochs = 100
#     # model_info = f"basic_model_{epochs}-epochs"
#     # model_info = "basic_model_cpu_1-epochs"
#     model_info = "sgd_model_10-epochs"
#     show_statistics(model_info)

# if __name__ == "__main__":
#     path_to_dataset_folder = "./data/filtered_dataset"
#     train_size = 0.8
#     batch_size = 4#16
#     num_classes = 7
#     # device = "cuda" if torch.cuda.is_available() else "cpu"
#     device = "cpu"
#     save_stats = True
#     epochs = 1
#     model_info = f"basic_model_{epochs}-epochs"

#     print(scipy.__version__)

#     # Load dataset
#     print("Loading dataset")
#     train_loader, test_loader = load_dataset(path_to_dataset_folder, train_size, batch_size)

#     # path_to_dataset_video_folder = "./data/filtered_dataset/test_imgs_for_video"
#     # train_loader, test_loader = load_dataset(path_to_dataset_video_folder, train_size=0.01, batch_size=batch_size)

#     print(f"Train loader size: {len(train_loader)}, Test loader size: {len(test_loader)}")
#     print(f"cca Train images: {len(train_loader)*batch_size}, cca Test images: {len(test_loader)*batch_size}")


#     # Initialize model, loss
#     model = MobileV3Small(num_classes=num_classes)
#     # for param in model.parameters():
#     #     param.data.fill_(0.0)
#     # model = MobileV3Small.from_pretrained()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
#     CE = torch.nn.CrossEntropyLoss(reduction="none", weight=None)

#     # print(model.state_dict()["last.bias"])

#     # Train and test model4#8#
#     print("Start model training with testisng")
#     train_model_with_test(model, train_loader, optimizer, CE, epochs=epochs, model_info=model_info, device_str=device, save_stats=save_stats)
#     print("Finished model training with testing")

#     print("Print model statistics")
#     print_stats(model_info)


    
