import os
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from dataset import data_divider
from pytorch_pretrained_vit import ViT


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_args():
    """
    Parses and returns command-line arguments for training the model.

    Returns:
        Namespace: A Namespace object containing parsed arguments with the following attributes:
            - root (str): Root directory of the dataset.
            - model (str): Model selection code ('a', 'b', 'c', etc.).
            - epochs (int): Number of epochs for training.
            - batch_size (int): Batch size for data loading.
            - num_workers (int): Number of workers for data loading.
            - logging (str): Directory for logging.
    """
    parser = ArgumentParser(description='train model')
    parser.add_argument('--root', '-r', type=str,
                        default='./dataset/',
                        help='root directory of dataset')
    parser.add_argument('--model', '-m', type=str, default='e', help='select any model from a,b,c,d,e')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', '-n', type=int, default=4, help='number of workers')
    parser.add_argument('--logging', '-l', type=str, default='logging', help='logging directory')
    parser.add_argument('--phase', '-p', type=str, default='test', help='train or test phase')
    args = parser.parse_args()
    return args


def train_model(model, train_loader, test_loader, model_type):
    """
    Trains the given model using the provided training and testing data loaders.
    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the testing dataset.
        model_type (str): Type of the model (used for saving model checkpoints).
    Returns:
        None
    """
    for param in model.parameters():
        param.requires_grad = True
    # for param in list(model.parameters())[:-5]:
    #     param.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    print('Training Starts.....\n')
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(train_loader)
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                # Inception-v3 returns outputs as (main_output, aux_output)
                main_output, aux_output = outputs
                loss1 = criterion(main_output, labels)
                loss2 = criterion(aux_output, labels)
                loss = loss1 + 0.4 * loss2  # Auxiliary loss with weight 0.4
            else:
                loss = criterion(outputs, labels)
            # loss = criterion(outputs, labels)
            progress_bar.set_description(
                'Epoch: {}/{} Iter: {} Loss: {:.4f}'.format(epoch + 1, epochs, iter + 1, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        all_predictions = []
        all_labels = []
        progress_bar2 = tqdm(test_loader)
        for iter, (images, labels) in enumerate(progress_bar2):
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)
                predictions = torch.argmax(outputs.cpu(), dim=1)
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu())
        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]
        acc = accuracy_score(all_labels, all_predictions)
        print('Epoch: {}/{} Test Loss: {:.4f} Test Acc: {:.4f}'.format(epoch + 1, epochs, loss.item(), acc))
        torch.save(model.state_dict(), f'model/last_{model_type}.pt')
        if acc > best_acc:
            torch.save(model.state_dict(), f'model/best_{model_type}.pt')
            best_acc = acc
    print('Training Completed!!!')

def test_model(model, test_loader, model_type):
    """
    Tests the given model using the provided test data loader and prints the accuracy.
    Args:
        model (nn.Module): The neural network model to test.
        test_loader (DataLoader): DataLoader for the testing dataset.
        model_type (str): Type of the model (used for loading the best model checkpoint).
    Returns:
        None
    """
    best_file_path = f'model/best_{model_type}.pt'
    if os.path.exists(best_file_path):
        model.load_state_dict(torch.load(best_file_path, map_location=torch.device('cpu')))
    best_model = model
    best_model.eval()
    all_predictions_best = []
    all_labels_best = []
    progress_bar = tqdm(test_loader)
    for iter, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = best_model(images)
            predictions = torch.argmax(outputs.cpu(), dim=1)
            all_predictions_best.extend(predictions)
            all_labels_best.extend(labels.cpu())
    all_labels_best = [label.item() for label in all_labels_best]
    all_predictions_best = [prediction.item() for prediction in all_predictions_best]
    report = classification_report(all_labels_best, all_predictions_best, output_dict=True)
    accuracy = report['accuracy']
    macro_avg = report['macro avg']
    weighted_avg = report['weighted avg']
    print(f'Accuracy : {accuracy:.4f}')
    print(f'Macro Average Precision: {macro_avg["precision"]:.4f}')
    print(f'Macro Average Recall: {macro_avg["recall"]:.4f}')
    print(f'Macro Average F1-Score: {macro_avg["f1-score"]:.4f}')
    print(f'Weighted Average Precision: {weighted_avg["precision"]:.4f}')
    print(f'Weighted Average Recall: {weighted_avg["recall"]:.4f}')
    print(f'Weighted Average F1-Score: {weighted_avg["f1-score"]:.4f}')
    print('Test Completed!!!')


def model_decleration_pretrained(model_code, output_feature=124):
    """
    Declares and returns a pretrained model based on the given model code.
    Args:
        model_code (str): Code representing the desired model:
                          - "a" for VGG16
                          - "b" for Vision Transformer (ViT)
                          - "c" for ResNet50
                          - "d" for GoogLeNet (Inception v1)
                          - "e" for EfficientNet-B0
        output_feature (int): Number of output features for the final classification layer. Default is 10.
    Returns:
        nn.Module: The specified pretrained model with the final layer modified for the given output features.
    """
    if model_code == "a":
        model_a = models.vgg16(pretrained=True)
        print('Pretrained vgg16-Model Loaded Successfully!')
        model_a.classifier[6] = nn.Linear(model_a.classifier[6].in_features, output_feature)
        model_a = model_a.to(device)
        return model_a
    elif model_code == "b":
        model_b = ViT('B_16_imagenet1k', pretrained=True, image_size=224, num_classes=output_feature).to(device)
        print('Pretrained ViT-Model Loaded Successfully!')
        return model_b
    elif model_code == "c":
        model_c = models.resnet50(pretrained=True)
        print('Pretrained resnet50-Model Loaded Successfully!')
        model_c.fc = nn.Linear(model_c.fc.in_features, output_feature)
        model_c = model_c.to(device)
        return model_c
    elif model_code == "d":
        model_d = models.googlenet(pretrained=True)
        print('Pretrained inception_v1-Model Loaded Successfully!')
        model_d.fc = nn.Linear(model_d.fc.in_features, output_feature)
        model_d = model_d.to(device)
        return model_d
    elif model_code == "e":
        model_e = models.efficientnet_b0(pretrained=True)
        print('Pretrained efficientnet_b0-Model Loaded Successfully!')
        model_e.classifier[1] = nn.Linear(model_e.classifier[1].in_features, output_feature)
        model_e = model_e.to(device)
        return model_e


if __name__ == '__main__':
    args = get_args()
    root = args.root
    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    logging = args.logging
    model_type = args.model
    # Split the dataset into train, validation, and test sets
    train_dataset, test_dataset = data_divider(args.root, args.batch_size, model_type)
    model = model_decleration_pretrained(model_type)
    if args.phase == 'train':
        train_model(model, train_dataset, test_dataset, model_type)
        print('Evaluating the performance...................')
    if args.phase == 'test':
        test_model(model, test_dataset, model_type)
        print('Finished!!!')
