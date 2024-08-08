import os
import time
from argparse import ArgumentParser
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from torchvision import models, transforms

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
    parser.add_argument('--model', '-m', type=str, default='', help='select any model from a,b,c')
    parser.add_argument('--epochs', '-e', type=int, default=2, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', '-n', type=int, default=4, help='number of workers')
    parser.add_argument('--logging', '-l', type=str, default='logging', help='logging directory')
    args = parser.parse_args()
    return args


def test_model(best_model_a, best_model_b, best_model_c, best_model_d, best_model_e, test_dataset):
    """
    Tests multiple models on the given test dataset and computes the accuracy using feature level fusion.

    Args:
        best_model_a (nn.Module): Pretrained model A.
        best_model_b (nn.Module): Pretrained model B.
        best_model_c (nn.Module): Pretrained model C.
        best_model_d (nn.Module): Pretrained model D.
        best_model_e (nn.Module): Pretrained model E.
        test_dataset (DataLoader): DataLoader for the testing dataset.

    Returns:
        None
    """
    best_model_a.eval()
    best_model_b.eval()
    best_model_c.eval()
    best_model_d.eval()
    best_model_e.eval()
    all_predictions_merged = []
    all_labels_merged = []
    for iterations, (images, labels) in enumerate(tqdm(test_dataset, desc="Testing")):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            # for model a
            outputs_a = best_model_a(images)
            # for model b
            outputs_b = best_model_b(images)
            # for model c
            outputs_c = best_model_c(images)
            # for model d
            outputs_d = best_model_c(images)
            # for model e
            outputs_e = best_model_c(images)
            # feature merging
            merged_feature = (outputs_a + outputs_b + outputs_c + outputs_d + outputs_e) / 5.0
            predictions_merged = torch.argmax(merged_feature.cpu(), dim=1)
            all_predictions_merged.extend(predictions_merged)
            all_labels_merged.extend(labels.cpu())

    all_labels_merged = [label.item() for label in all_labels_merged]
    all_predictions_merged = [prediction.item() for prediction in all_predictions_merged]
    report = classification_report(all_labels_merged, all_predictions_merged, output_dict=True)
    accuracy = report['accuracy']
    macro_avg = report['macro avg']
    weighted_avg = report['weighted avg']
    print(f'Accuracy of Feature Level Fusion: {accuracy:.4f}')
    print(f'Macro Average Precision: {macro_avg["precision"]:.4f}')
    print(f'Macro Average Recall: {macro_avg["recall"]:.4f}')
    print(f'Macro Average F1-Score: {macro_avg["f1-score"]:.4f}')
    print(f'Weighted Average Precision: {weighted_avg["precision"]:.4f}')
    print(f'Weighted Average Recall: {weighted_avg["recall"]:.4f}')
    print(f'Weighted Average F1-Score: {weighted_avg["f1-score"]:.4f}')


def model_loading(output_feature=124):
    """
    Loads pretrained models and their checkpoints, modifying the final layer for the given number of output features.

    Args:
        output_feature (int): Number of output features for the final classification layer. Default is 10.

    Returns:
        tuple: A tuple containing the loaded models (model_a, model_b, model_c, model_d, model_e).
    """
    model_a = models.vgg16(pretrained=True)
    model_a.classifier[6] = nn.Linear(model_a.classifier[6].in_features, output_feature)
    model_a = model_a.to(device)
    if os.path.exists('model/best_a.pt'):
        model_a.load_state_dict(torch.load('./model/best_a.pt', map_location=torch.device('cpu')))
        print('Model(a) vgg16 and Checkpoint Loaded Successfully!')
    model_b = ViT('B_16_imagenet1k', pretrained=True, image_size=224, num_classes=output_feature).to(device)
    if os.path.exists('model/best_b.pt'):
        model_b.load_state_dict(torch.load('./model/best_b.pt', map_location=torch.device('cpu')))
        print('Model(b) ViT and Checkpoint Loaded Successfully!')
    model_c = models.resnet50(pretrained=True)
    model_c.fc = nn.Linear(model_c.fc.in_features, output_feature)
    model_c = model_c.to(device)
    if os.path.exists('model/best_c.pt'):
        model_c.load_state_dict(torch.load('./model/best_c.pt', map_location=torch.device('cpu')))
        print('Model(c) resnet50 and Checkpoint Loaded Successfully!')
    model_d = models.googlenet(pretrained=True)
    model_d.fc = nn.Linear(model_d.fc.in_features, output_feature)
    model_d = model_d.to(device)
    if os.path.exists('model/best_d.pt'):
        model_d.load_state_dict(torch.load('./model/best_d.pt', map_location=torch.device('cpu')))
        print('Model(d) inception_v1 and Checkpoint Loaded Successfully!')
    model_e = models.efficientnet_b0(pretrained=True)
    model_e.classifier[1] = nn.Linear(model_e.classifier[1].in_features, output_feature)
    model_e = model_e.to(device)
    if os.path.exists('model/best_e.pt'):
        model_e.load_state_dict(torch.load('./model/best_e.pt', map_location=torch.device('cpu')))
        print('Model(e) efficientnet_b0 and Checkpoint Loaded Successfully!')
    return model_a, model_b, model_c, model_d, model_e


if __name__ == '__main__':
    args = get_args()
    root = args.root
    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    logging = args.logging
    model_type = args.model
    train_dataset, test_dataset = data_divider(args.root, args.batch_size, model_type)
    model_a, model_b, model_c, model_d, model_e = model_loading()
    print('Evaluating the performance...................')
    test_model(model_a, model_b, model_c, model_d, model_e, test_dataset)
