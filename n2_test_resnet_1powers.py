#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol
import numpy as np
import pandas as pd
import torch

def test_model_regression(totalloader, net, classes, device):
    total_n2 = 0
    total_power = 0

    n2_classes = classes['n2']
    
    correct_pred_n2 = {classname: 0 for classname in n2_classes}
    total_pred_n2 = {classname: 0 for classname in n2_classes}

    squared_errors_power = []

    with torch.no_grad():
        for images, powers, labels_n2 in totalloader:
            images = images.to(device)
            powers = powers.to(device)
            labels_n2 = labels_n2.to(device)

            outputs_n2, outputs_power = net(images)
            _, predicted_n2 = torch.max(outputs_n2, 1)
            outputs_power = outputs_power.flatten()

            total_n2 += labels_n2.size(0)
            total_power += powers.size(0)
            
            # Update counts for n2
            for label, prediction in zip(labels_n2, predicted_n2):
                label_name = n2_classes[label.item()]
                correct_pred_n2[label_name] += (prediction == label).item()
                total_pred_n2[label_name] += 1

            # Calculate squared errors for power
            squared_errors_power.extend((powers - outputs_power.squeeze()).pow(2).tolist())

    # n2 accuracy
    correct_n2 = sum(correct_pred_n2.values())
    print(f"\nAccuracy for 'n2' predictions: {100 * correct_n2 / total_n2:.2f}%")
    for classname, correct_count in correct_pred_n2.items():
        accuracy = 100 * correct_count / total_pred_n2[classname] if total_pred_n2[classname] > 0 else 'N/A'
        print(f"Accuracy for class '{classname}' in 'n2': {accuracy}%")

    # Mean squared error for power
    mse_power = np.mean(squared_errors_power)
    print(f'\nMean Squared Error for power: {mse_power:.4f}')

def test_model_classification(totalloader, net, classes, device, backend):
    correct_n2 = 0
    total = 0

    n2_classes = classes['n2']

    correct_pred_n2 = {classname: 0 for classname in n2_classes}
    total_pred_n2 = {classname: 0 for classname in n2_classes}

    distance_errors_n2 = []
    percentage_errors_n2 = []

    with torch.no_grad():
        for images, powers, powers_labels, labels in totalloader:
            if backend == "GPU":
                images = images
                labels_power = powers_labels
                powers_values = torch.from_numpy(powers.cpu().numpy()[:,np.newaxis]).float().to(device)
                labels_n2 = labels
            else:
                images = images.to(device) 
                labels_power = powers_labels.to(device) 
                powers_values = torch.from_numpy(powers.numpy()[:,np.newaxis]).float().to(device)
                labels_n2 = labels.to(device)

            outputs_n2 = net(images, powers_values)
            _, predicted_n2 = torch.max(outputs_n2, 1)


            total += labels_n2.size(0)
            correct_n2 += (predicted_n2 == labels_n2).sum().item()

            # Update counts and distance errors for n2
            for label, prediction in zip(labels_n2, predicted_n2):
                label_name = n2_classes[label.item()]
                correct_pred_n2[label_name] += (prediction == label).item()
                total_pred_n2[label_name] += 1
                if prediction.item() != label.item():
                    distance_errors_n2.append(abs(label.item() - prediction.item()))
                    if float(n2_classes[prediction.item()]) < float(n2_classes[label.item()]):
                        percentage_errors_n2.append( np.abs((float(n2_classes[prediction.item()]) - float(n2_classes[label.item()]))*100/float(n2_classes[label.item()])))
                    else:
                        percentage_errors_n2.append( np.abs((float(n2_classes[prediction.item()]) - float(n2_classes[label.item()]))*100/float(n2_classes[prediction.item()])))

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

def count_parameters_pandas(model):
    data = []
    total_params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        data.append([name, params])
        total_params += params

    df = pd.DataFrame(data, columns=["Modules", "Parameters"])
    print(df)
    print(f"Total Trainable Params: {total_params}")
    return total_params

# if __name__ == "__main__":
#     from n2_finder_resnet_10powers import data_split, data_treatment
#     from model_resnetv2 import Inception_ResNetv2

#     path = "/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN"
#     resolution = 512
#     number_of_n2 = 10
#     number_of_puiss = 10

#     print("---- DATA LOADING ----")
#     # file = f'{path}/Es_w{resolution}_n2{number_of_n2}_puiss{number_of_puiss}_real_imag_out_extended_noise.npy' #10 x 10
#     file = f'{path}/Es_w{resolution}_n2{number_of_n2}_puiss{number_of_puiss}_at0.02_real_imag_out_extended_noise.npy' # 10 x 1
    
#     E_noisy = np.load(file)

#     n2_label_clean = np.tile(np.arange(0, number_of_n2), number_of_puiss)
#     n2_label_noisy = np.repeat(n2_label_clean, 23)
#     puiss_label_clean = np.repeat(np.arange(0, number_of_puiss), number_of_n2)
#     puiss_value_clean = np.repeat(np.linspace(0.02, .5, number_of_puiss), number_of_n2)
#     puiss_label_noisy = np.repeat(puiss_label_clean, 23)
#     puiss_value_noisy = (np.repeat(puiss_value_clean, 23) - 0.02 ) / (1 - 0.02)

#     classes = {
#         'n2': tuple(map(str, np.linspace(-1e-9, -1e-10, number_of_n2))),
#         'power' : tuple(map(str, np.linspace(0.02, 1, number_of_puiss)))
#     }
#     batch_size = 6

#     backend = "GPU"
#     if backend == "GPU":
#         device = torch.device("cuda:0")
#     else:
#         device = torch.device("cpu")

#     print("---- DATA TREATMENT ----")
#     train_set, validation_set, test_set = data_split(E_noisy,n2_label_noisy, puiss_label_noisy, puiss_value_noisy, 0.8, 0.1, 0.1)

#     train, train_n2_label, train_puiss_label,train_puiss_value = train_set
#     validation, validation_n2_label, validation_puiss_label,validation_puiss_value = validation_set
#     test, test_n2_label, test_puiss_label,test_puiss_value = test_set

#     training_test = False
#     totalloader = data_treatment(test, test_n2_label, test_puiss_label,test_puiss_value, batch_size, device,training_test )

#     print("---- MODEL LOADING ----")
#     cnn = Inception_ResNetv2(in_channels=E_noisy.shape[1], class_n2=number_of_n2, class_power=number_of_puiss, batch_size=batch_size)
#     # cnn = nn.DataParallel(cnn, device_ids=[0, 1])
#     cnn = cnn.to(device)
#     date = "21-03-2024_140949_training"
#     cnn.load_state_dict(torch.load(f'{path}/{date}/n2_net_w{resolution}_n2{number_of_n2}_puiss{number_of_puiss}_2D.pth'))

#     print("---- MODEL ANALYSIS ----")
#     count_parameters_pandas(cnn)

#     print("---- MODEL TESTING ----")
#     test_model_classification(totalloader, cnn, classes, device, backend)