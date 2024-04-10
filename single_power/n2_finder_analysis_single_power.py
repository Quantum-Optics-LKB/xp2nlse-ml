#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol
from tqdm import tqdm
import pandas as pd
import numpy as np

def analysis(path: str, power_values: np.ndarray):
    """
    Analyzes and aggregates the accuracy and index error across different power values for various 
    models and data type configurations, ultimately saving the aggregated results in a JSON file.

    This function cycles through a list of predefined data types and model versions, reading testing 
    results for each combination of power value, model version, and data type. It extracts accuracy 
    and index error metrics from the 'testing.txt' files located in respective subdirectories. These 
    metrics are then compiled into a structured format that associates each set of metrics with its 
    corresponding model version, data type, and power value, exporting the aggregated analysis as a 
    JSON file for further exploration.

    Parameters:
    - path (str): The directory path where model training results are stored, expected to contain 
      subdirectories for each model version and data type configuration.
    - power_values (np.ndarray): A list of power values for which the models have been evaluated. This list 
      is used to locate the specific 'testing.txt' files within each subdirectory.

    The function iterates over multiple data types and model versions, using the provided list of power 
    values to access and process specific testing results for each configuration. The aggregated results 
    are saved in a JSON file named 'model_analysis_single_power.json' within the specified path.

    Note: The function assumes a specific structure for the 'testing.txt' files and expects the accuracy 
    and index error information to be formatted consistently across these files. The presence of the 
    specific "TESTING" line in the files is used as a marker to locate the relevant metrics.
    """
    data_types = ["amp", "amp_pha", "amp_pha_unwrap", "pha", "pha_unwrap", "amp_pha_pha_unwrap"]
    data_type_results = {}
    for data_types_index in range(len(data_types)):

        models = {}
        for model_index in range(2, 6):
            
            model_version =  f"model_resnetv{model_index}_1powers"
            model = {}
            power_list_accuracy_n2 = []
            power_list_index_error_n2 = []

            power_list_accuracy_isat = []
            power_list_index_error_isat = []
            for power in tqdm(power_values, position=4,desc="Iteration", leave=False):
                
                stamp = f"power{str(power)[:4]}_{data_types[data_types_index]}_{model_version}"
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
                error = lines[count+13].split(" ")[-1].split("\n")[0]
                power_list_accuracy_n2.append(float(accuracy))
                if len(accuracy) == len("100.00"):
                    power_list_index_error_n2.append(0)
                else:
                    power_list_index_error_n2.append(float(error))
                
                accuracy = lines[count+16].split(" ")[-1].split("%")[0]
                error = lines[count+27].split(" ")[-1].split("\n")[0]
                power_list_accuracy_isat.append(float(accuracy))
                if len(accuracy) == len("100.00"):
                    power_list_index_error_isat.append(0)
                else:
                    power_list_index_error_isat.append(float(error))
                f.close
                model["accuracy_n2"] = power_list_accuracy_n2
                model["index_error_n2"] = power_list_index_error_n2 

                model["accuracy_isat"] = power_list_accuracy_isat
                model["index_error_isat"] = power_list_index_error_isat 
                
            models[model_version] = model
        data_type_results[data_types[data_types_index]] = models

    df = pd.DataFrame.from_dict(data_type_results, orient="columns")
    df.to_json(f'{path}/model_analysis_single_power.json')            
                
if __name__ == "__main__":
    path ="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN"
    power_values = np.linspace(0.02, 0.5001, 10)
    analysis(path, power_values)
