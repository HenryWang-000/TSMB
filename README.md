# Rerunning the experiments

* This project is based on Python version 3.8.10

## Obtaining optimal results for time delay on different datasets and bootstrap-based time delay optimization results

Open a command prompt window in the Windows environment and execute the following command:

`python opt.py occupancy fixed cc bootstrap 1min 0.25 100`

Args:

- `occupancy`: A parameter for the data_set, referring to the DataSetName class in enums.py.
- `fixed`: A parameter for data_arrange, referring to the DataArrange class in enums.py.
- `cc`: A parameter for method_name, referring to the CorrMethod class in enums.py.
- `bootstrap`: A parameter for opt_name, referring to the OptType class in enums.py.
- `1min`: A parameter for tolerance, referring to the Tolerance class in enums.py.
- `0.25`: A parameter for drop_rate, representing the proportion of block-bootstrap division of data, only required when data_arrange is set to bootstrap.
- `100`: A parameter for nums, representing the number of bootstrap iterations, only required when data_arrange is set to bootstrap.

## TSMB modeling

At the beginning of the file, define relevant variables to specify runtime parameters and configurations.

Import the corresponding dataset and the file containing the optimal results for single-time-delay optimization obtained in the previous step.

Run `tsmb_model.py` to obtain the modeling results under single-time-delay optimization and bootstrap-based time delay optimization.

## Generating perturbed time delays

At the beginning of the file, define relevant variables to specify runtime parameters and configurations.

Import the corresponding dataset and the file containing the optimal results for single-time-delay optimization obtained in the previous step.

Run `generate_perturbed_time_delay.py` to obtain the perturbed time delay file.

## Perturbed modeling

At the beginning of the file, define relevant variables to specify runtime parameters and configurations.

Import the corresponding dataset and the perturbed time delay file.

Run `perturbed_model.py` to obtain the modeling results based on perturbed time delays.

## TFT modeling

At the beginning of the file, define relevant variables to specify runtime parameters and configurations.

Import the corresponding dataset and the files containing the optimal results for single-time-delay and bootstrap-based time delay optimization obtained in the previous step.

Run `tft_model.py` to obtain the TFT modeling results.

## Generating final data

At the beginning of the file, define relevant variables to specify runtime parameters and configurations.

Run `results_to_csv.py` to obtain result files for each dataset.

Run `merge_results_csv.py` to merge the result files of each dataset for easier plotting.

## Plotting

Run `plot.ipynb` to obtain the plotting results based on the final data.



# DataSets:

The mineral dataset, which belongs to a real grinding process in a mine, is not displayed to prevent data leakage. The descriptions and download links for the other datasets are as follows:

- `Occupancy Detection`: <https://archive.ics.uci.edu/dataset/357/occupancy+detection>
- `pump_sensor_data`: <https://www.kaggle.com/datasets/nphantawee/pump-sensor-data>
- `ItalyPowerDemand`: <https://www.timeseriesclassification.com/description.php?Dataset=ItalyPowerDemand>
- `Air Quality`: <https://archive.ics.uci.edu/dataset/360/air+quality>