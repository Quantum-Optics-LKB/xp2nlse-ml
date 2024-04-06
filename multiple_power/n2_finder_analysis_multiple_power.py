#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol
import pandas as pd

def analysis(path: str):
    """
    Analyzes and aggregates the accuracy and index error of different models based on their 
    data type configurations, saving the results in a JSON file.

    This function iterates over predefined data types and model versions, reading accuracy and index 
    error metrics from stored testing results. It compiles these metrics into a structured format, 
    associating them with their respective model and data type, and then exports the aggregated 
    analysis as a JSON file for further examination.

    Parameters:
    - path (str): The directory path where model training results are stored. This path is expected to 
      contain subdirectories for each model version and data type configuration, each with a 'testing.txt' 
      file that includes the accuracy and index error information.

    The function focuses on multiple pre-specified data types and iterates through a range of model versions. 
    For each model version and data type, it reads the testing results, extracts relevant metrics, and organizes 
    them by model version and data type. Finally, it saves the aggregated results in a JSON file named 
    'model_analysis_multi_power.json' in the specified path.

    Note: The function relies on a specific structure of the 'testing.txt' file and assumes that the accuracy 
    and index error information is formatted in a consistent and predictable manner across these files.
    """
    data_types = ["amp", "amp_pha", "amp_pha_unwrap", "pha", "pha_unwrap", "amp_pha_pha_unwrap"]
    data_type_results = {}
    for data_types_index in range(len(data_types)):

        models = {}
        for model_index in range(2, 6):
            
            model_version =  f"model_resnetv{model_index}_1powers"
            model = {}
            power_list_accuracy = []
            power_list_index_error = []
            stamp = f"multi_power_{data_types[data_types_index]}_{model_version}"
            new_path = f"{path}/{stamp}_training"

            f = open(f'{new_path}/testing.txt', 'r')

            count = 0
            for line in f.readlines():
                if "TESTING" in line:
                    break
                count += 1
            f.close()

            f = open(f'{new_path}/testing.txt', 'r')
            lines = f.readlines()
            accuracy = lines[count+2].split(" ")[-1].split("%")[0]
            power_list_accuracy.append(float(accuracy))
            if len(accuracy) == len("100.00"):
                power_list_index_error.append(0)
            else:
                power_list_index_error.append(float(lines[count+13].split(" ")[-1].split("\n")[0]))
            f.close
        model["accuracy"] = power_list_accuracy
        model["index_error"] = power_list_index_error 
            
    models[model_version] = model
    data_type_results[data_types[data_types_index]] = models

    df = pd.DataFrame.from_dict(data_type_results, orient="columns")
    df.to_json(f'{path}/model_analysis_multi_power.json')            