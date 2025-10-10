# Explorations on Distribution Shift for Hypotension and Hypoxemia Prediction on MIMIC III matched waveform v.1 


## Setup
1. Install the required dependencies from requiremens.txt via ```pip intsall -r requirements.txt```

## Repository structure

Top-level layout of this repository (important files and directories):

- `clean_data_final.ipynb` - Notebook used for final cleaning/processing of the datasets.
- `ReadMe.md` - This file. Describes the project, setup and how to reproduce experiments.
- `requirements.txt` - Python dependencies used by the project.
- `data.zip` - (Optional) compressed archive of raw or processed data used for quick sharing.
- `cohort_creation/` - Scripts and notebooks used to create patient cohorts and helper utilities for dataset creation and exploration.
- `configs/` - Experiment and dataset creation configuration files (e.g. JSON/YAML) used by training and evaluation scripts.
- `data/` - Preprocessed dataset files (PyTorch tensors and other artifacts) ready for training and testing.
- `distribution_shift/` - Code and experiments related to studying distribution shift across cohorts and settings.

Use the folders above as a starting point to find dataset creation scripts, experiment configs, and preprocessed data for running the training and evaluation pipelines.



## Reproducibility

Unzip all zipped folders.

### How to create datasets

#### Filter for valid recording times for the cohorts 
You can skip the filtering step, as the results of it are provided in [.\configs\dataset_creation_confgis\inputs](.\configs\dataset_creation_confgis\inputs).
A rougher visual outline of the cohort creation/filering steps is provided in [.\cohort_creation\visual_documentation\](.\cohort_creation\visual_documentation\).

1. The MIMIC III matched waveform dataset was explored via [./cohort_creation/exploration_mimic_iii_matched_waveform.ipynb](./cohort_creation/exploration_mimic_iii_matched_waveform.ipynb), this also creates the `.\cohort_creation\data\mimic2wdb-matched_numerics_signals_duration.csv` which is used by [.\cohort_creation\mimic_waveform_explore_helper.py](.\cohort_creation\mimic_waveform_explore_helper.py) to quickly filter recordings by theyr duration.
2.  Run [./cohort_creation/hinrichs_base_model_dataset_creation.ipynb](./cohort_creation/hinrichs_base_model_dataset_creation.ipynb) - this will create the file [./cohort_creation/data/hinrichs_dataset/valid_records_hinrichs_base_model.json](./cohort_creation/data/hinrichs_dataset/valid_records_hinrichs_base_model.json) - which contains all records that meet the initial selection criteria: recordings >= 90 minutes and have recordings for CVP, HR, NBP Sys, NBPDias, NBPMean and SpO2.
3. To create the individual cohorts as defined in the paper, run the corresponding `./cohort_creation/cohort_creation_xxx.py` file. This will create two `./cohort_creation/data/hinrichs_dataset/records_xxxx.txt` files, which correspond to the record_ids of records that fullfill the criteria for the xxx and no_xxx cohort. Finally two `./cohort_creation/data/records_with_start_endtime/xxx.csv` and `./cohort_creation/data/records_with_start_endtime/no_xxx.csv` are created, these files contain the record_id, as well as the start and end offset of the usable time of the record. 

#### Create the datasets based on the results of filtering

The created datasets will be provided upon reasonable request, they were not uploaded due to file-size restriction.

The `./cohort_creation/data/records_with_start_endtime/xxx.csv` and `./cohort_creation/data/records_with_start_endtime/no_xxx.csv` are the input for our next step: the actual dataset creation step.

1. Copy the folders `configs` and `inputs` from [.\configs\dataset_creation_configs](.\configs\dataset_creation_configs) to [Algorithm2Domain/Datasets/mimic-iii-matched-waveform-dataset-creation/configs](https://gitlab.com/BI_Koeln/Algorithm2Domain/-/tree/main/Datasets/mimic-iii-matched-waveform-dataset-creation/configs?ref_type=heads)
2. The exact steps to create the datasets via the `mimic-iii-matched-waveform-dataset-creation` scripts are documented in the powershells in [.\configs\dataset_creation_powershell_scripts](.\configs\dataset_creation_powershell_scripts)
3. The datasets were cleaned a final time using the [.\clean_data_final.ipynb](.\clean_data_final.ipynb) - all records containing at least one item outside of channel specific thresholds were removed.


### How to rerun experiments

Results of the experiments can be found in [.\distribution_shift\results\CNN_NO_ADAPT](.\distribution_shift\results\CNN_NO_ADAPT)

All experiments have been conducted with `ALGORITHM2DOMAINAAAAAAAAAAAAAAAAAAAAAA`.
Please find the sweep-configuration in [./configs/no_adapt_cnn_experiment_configs/sweep_configs](./configs/no_adapt_cnn_experiment_configs/sweep_configs). The configurations of the best performing and selected models for each task can be found in the respective folders in [./configs/no_adapt_cnn_experiment_configs/](./configs/no_adapt_cnn_experiment_configs/).
If you want to rerun the experiments, please copy those files to `.\configs` of [Algorihm2Domain_AdaTime](https://gitlab.com/BI_Koeln/Algorithm2Domain/-/tree/main/Evaluation_Framework/Algorithm2Domain_AdaTime/configs?ref_type=heads).

Run the hyperparameter tuning e.g. via the command:
```
. ./venv/bin/activate
python main_sweep.py --dataset PHD --da_method NO_ADAPT --backbone CNN --num_runs 1 --num_sweeps 16 --device cuda:0

```

Run the individual experiments e.g. via the command:
```
python main.py --phase train --exp_name vasopressors_cleaned_spo2_to_no_vasopressors_cleaned_spo2 --da_method NO_ADAPT --dataset PHD --backbone CNN
python main.py --phase test --exp_name vasopressors_cleaned_spo2_to_no_vasopressors_cleaned_spo2 --da_method NO_ADAPT --dataset PHD --backbone CNN
python main.py --phase all --exp_name vasopressors_cleaned_spo2_to_no_vasopressors_cleaned_spo2 --da_method NO_ADAPT --dataset PHD --backbone CNN
```


### How to measure the distribution shift and set it in relation to the models performance

1. Calculate the distribution shifts using [.\distribution_shift\distance_measures.ipynb](.\distribution_shift\distance_measures.ipynb) - currently KS Stat and KL divergence are supported.
2. Create graphs that set the distribution shifts into context of the model performance on sourc test and target via [.\distribution_shift\analysis_cnn_results.ipynb](.\distribution_shift\analysis_cnn_results.ipynb)

Bonus: If you want to visually insepct the domain adaptation scenarios use [.\distribution_shift\data_visualization.ipynb](.\distribution_shift\data_visualization.ipynb)



## References 

We used the MIMIC III matched waveform v1.0

Moody, B., Moody, G., Villarroel, M., Clifford, G. D., & Silva, I. (2020). MIMIC-III Waveform Database Matched Subset (version 1.0). PhysioNet. RRID:SCR_007345. [https://doi.org/10.13026/c2294b](https://doi.org/10.13026/c2294b)

[Johnson, A. E. W., Pollard, T. J., Shen, L., Lehman, L. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L. A., & Mark, R. G. (2016). MIMIC-III, a freely accessible critical care database. Scientific Data, 3, 160035.](https://www.nature.com/articles/sdata201635)

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.


## Citation

TODO