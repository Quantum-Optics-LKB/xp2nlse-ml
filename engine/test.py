#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol
import numpy as np
import torch

def test_model_classification(totalloader, net, classes, device):
    """
    Tests the classification accuracy of a trained neural network model on a given dataset.

    Parameters:
    - totalloader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
    - net (torch.nn.Module): The trained neural network model to be evaluated.
    - classes (dict): A dictionary with keys 'n2' and 'isat', where each key maps to a list of class names
      corresponding to the output predictions of the network.

    The function iterates over the test dataset to evaluate the model's performance on n2 and isat 
    classification tasks, computing overall accuracy, class-wise accuracy, average distance error 
    (how far the predicted class is from the true class), and average percentage error for misclassifications.

    Outputs are printed to the console, summarizing the model's classification performance and error metrics.
    """
    correct_n2 = 0
    correct_isat = 0
    total = 0

    n2_classes = classes['n2']
    isat_classes = classes['isat']

    correct_pred_n2 = {classname: 0 for classname in n2_classes}
    total_pred_n2 = {classname: 0 for classname in n2_classes}

    correct_pred_isat = {classname: 0 for classname in isat_classes}
    total_pred_isat = {classname: 0 for classname in isat_classes}

    distance_errors_n2 = []
    distance_errors_isat = []

    percentage_errors_n2 = []
    percentage_errors_isat = []

    with torch.no_grad():
        for images, n2_labels, isat_labels in totalloader:
            # Process original images
            images = images.to(device)
            n2_labels = n2_labels.to(device)
            isat_labels = isat_labels.to(device)

            outputs_n2, outputs_isat = net(images)
            _, predicted_n2 = torch.max(outputs_n2, 1)
            _, predicted_isat = torch.max(outputs_isat, 1)

            total += n2_labels.size(0)
            correct_n2 += (predicted_n2 == n2_labels).sum().item()
            correct_isat += (predicted_isat == isat_labels).sum().item()

            # Update counts and distance errors for n2
            for label, prediction in zip(n2_labels, predicted_n2):
                label_name = n2_classes[label.item()]
                correct_pred_n2[label_name] += (prediction == label).item()
                total_pred_n2[label_name] += 1
                if prediction.item() != label.item():
                    distance_errors_n2.append(abs(label.item() - prediction.item()))
                    if float(n2_classes[prediction.item()]) < float(n2_classes[label.item()]):
                        percentage_errors_n2.append( np.abs((float(n2_classes[prediction.item()]) - float(n2_classes[label.item()]))*100/float(n2_classes[label.item()])))
                    else:
                        percentage_errors_n2.append( np.abs((float(n2_classes[prediction.item()]) - float(n2_classes[label.item()]))*100/float(n2_classes[prediction.item()])))

            for label, prediction in zip(isat_labels, predicted_isat):
                label_name = isat_classes[label.item()]
                correct_pred_isat[label_name] += (prediction == label).item()
                total_pred_isat[label_name] += 1
                if prediction.item() != label.item():
                    distance_errors_isat.append(abs(label.item() - prediction.item()))
                    if float(isat_classes[prediction.item()]) < float(isat_classes[label.item()]):
                        percentage_errors_isat.append( np.abs((float(isat_classes[prediction.item()]) - float(isat_classes[label.item()]))*100/float(isat_classes[label.item()])))
                    else:
                        percentage_errors_isat.append( np.abs((float(isat_classes[prediction.item()]) - float(isat_classes[label.item()]))*100/float(isat_classes[prediction.item()])))

    print(f"\nAccuracy for 'n2' predictions: {100 * correct_n2 / total:.2f}%")
    for classname in n2_classes:
        accuracy = 100 * correct_pred_n2[classname] / total_pred_n2[classname] if total_pred_n2[classname] > 0 else 'N/A'
        print(f"Accuracy for class {classname} in 'n2': {accuracy}%")

    if distance_errors_n2:
        avg_distance_error_n2 = np.mean(distance_errors_n2)
        print(f'Average distance error for n2: {avg_distance_error_n2:.2f}')
    else:
        print('Average distance error for n2: N/A (no incorrect predictions)')

    if percentage_errors_n2:
        avg_percentage_error_n2 = np.mean(percentage_errors_n2)
        print(f'Average percentage error for n2: {avg_percentage_error_n2:.2f}%')
    else:
        print('Average percentage error for n2: N/A (no incorrect predictions)')


    print(f"\nAccuracy for 'isat' predictions: {100 * correct_isat / total:.2f}%")
    for classname in isat_classes:
        accuracy = 100 * correct_pred_isat[classname] / total_pred_isat[classname] if total_pred_isat[classname] > 0 else 'N/A'
        print(f"Accuracy for class {classname} in 'isat': {accuracy}%")

    if distance_errors_isat:
        avg_distance_error_isat = np.mean(distance_errors_isat)
        print(f'Average distance error for isat: {avg_distance_error_isat:.2f}')
    else:
        print('Average distance error for isat: N/A (no incorrect predictions)')

    if percentage_errors_isat:
        avg_percentage_error_isat = np.mean(percentage_errors_isat)
        print(f'Average percentage error for isat: {avg_percentage_error_isat:.2f}%')
    else:
        print('Average percentage error for isat: N/A (no incorrect predictions)')

def count_parameters_pandas(model):
    """
    Counts the total number of trainable parameters in a neural network model and prints a summary.

    Parameters:
    - model (torch.nn.Module): The neural network model whose parameters are to be counted.

    This function iterates through all trainable parameters of the given model, summarizing the count
    of parameters per module and the total count of trainable parameters across the model. The summary
    is printed in a tabular format using pandas DataFrame for clear visualization.

    Returns:
    - total_params (int): The total number of trainable parameters in the model.

    Example usage:
    - total_trainable_params = count_parameters_pandas(my_model)
      Prints a table summarizing parameters per module and the total trainable parameters.
    """
    data = []
    total_params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        data.append([name, params])
        total_params += params

    print(f"Total Trainable Params: {total_params}")
    return total_params