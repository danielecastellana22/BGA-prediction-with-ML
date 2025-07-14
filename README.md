# Predictive Modeling of Biogeographical Ancestry using a novel SNP panel and Supervised Learning approaches

This repository contains the official code of the paper *Predictive Modeling of Biogeographical Ancestry using a 
novel SNP panel and Supervised Learning approaches* by Grazzini et al.

If you use this code in your research, please cite:

```
@article{grazzini2025,
    title = {Predictive Modeling of Biogeographical Ancestry using a novel SNP panel and Supervised Learning 
    approaches},
    author = {Cosimo Grazzinia and Giorgia Spera and Stefania Morelli and Daniele Castellana and Giulia Cosenza 
    and Michela Baccini and Giulia Cereda and Elena Pilli},
    year = {2025},
    journal = {Experts Systems with Applications}
    note = {Under review}
}
```

## Project Structure

- `parse_data.py`: Processes the raw CSV data and creates a pickle file
- `clean_data.py`: Handles data cleaning, missing values, and encoding
- `create_splits.py`: Creates train/validation/test splits using stratified k-fold
- `config/`: Contains the configuration files used for the grid search
- `utils.py`: Contains utility functions and helper methods used across the project
- `run_test_eval.py`: Executes model evaluation on test datasets and generates performance metrics
- `run_model_selection.py`: Performs hyperparameter tuning and model selection using grid search
- `run_collect_results.py`: Aggregates and summarizes results from model evaluation and selection

## Data Preprocessing

- Handles missing values (removes columns with >3.5% missing data)
- Performs label encoding for regions
- Creates separate datasets for Giulia regions and continental regions
- Splits data into train/validation/test sets using stratified k-fold

## Model Configuration

The `config/` directory contains configuration files used for hyperparameter tuning during the grid search process. Each
machine learning model evaluated in this project has its dedicated configuration file that specifies:

- The range of hyperparameters to be evaluated
- Model-specific settings and options
- Search space for optimization parameters


