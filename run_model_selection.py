import os
os.environ['OMP_NUM_THREADS'] = '1'
import time
import yaml
import numpy as np
from argparse import ArgumentParser
from itertools import product
from functools import reduce
from operator import mul
import os.path as osp
from concurrent.futures import ProcessPoolExecutor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm
from utils import *


def run_single_config(data_path, current_split_dir, output_dir, estimator, hyper_params_config, do_oversampling, do_one_hot):

    results_file = osp.join(output_dir, 'results.npz')

    if osp.exists(results_file):
        eprint(f'SKIPPED: {output_dir}')
        return

    # carico gli split
    train_idx, val_idx, test_idx = get_split_idx(current_split_dir)

    # LOAD the data
    X = np.load(osp.join(data_path, 'X.npy'))
    y = np.load(osp.join(data_path, 'y.npy'))
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    # Pre-processing
    preprocessing_steps = []
    preprocessing_steps.append(SimpleImputer(missing_values=-1, strategy="most_frequent", keep_empty_features=False))

    if do_one_hot:
        preprocessing_steps.append(OneHotEncoder(handle_unknown='ignore'))

    preprocessing_pipeline = make_pipeline(*preprocessing_steps)
    X_tr = preprocessing_pipeline.fit_transform(X_tr)

    # Check if oversampling
    if do_oversampling:
        ros = RandomOverSampler(random_state=42)
        X_tr_os, y_tr_os = ros.fit_resample(X_tr, y_tr)
    else:
        X_tr_os, y_tr_os = X_tr, y_tr

    # call and fit the model
    m = estimator(**hyper_params_config)  # SVC(C=0.1,...)

    # Calculate the execution time
    start_time = time.time()
    m.fit(X_tr_os, y_tr_os)

    X_val = preprocessing_pipeline.transform(X_val)
    X_test = preprocessing_pipeline.transform(X_test)

    ypred_tr = m.predict(X_tr)
    ypred_val = m.predict(X_val)
    ypred_test = m.predict(X_test)

    try:
        proba_tr = m.predict_proba(X_tr)
        proba_val = m.predict_proba(X_val)
        proba_test = m.predict_proba(X_test)
    except Exception:
        proba_tr = proba_val = proba_test = None

    end_time = time.time()
    elapsed_time = end_time - start_time

    results_dict = {'elapsed_time': elapsed_time,
                    'ypred_tr': ypred_tr, 'ypred_val': ypred_val, 'ypred_test': ypred_test}

    if proba_tr is not None:
        results_dict.update({'proba_tr': proba_tr, 'proba_val': proba_val, 'proba_test': proba_test})

    np.savez(results_file, **results_dict)


def parse_config_file(data_path, splits_path, results_path, config_filename):
    with open(config_filename, 'r') as f:
        config_dict = yaml.safe_load(f)

    estimator = string2class(config_dict['estimator'])
    oversampling = config_dict['oversampling']
    one_hot = config_dict['one_hot']
    grid_dict = config_dict['hyperparams_grid']
    # stor the path of the experiment
    config_dict['data_path'] = data_path
    config_dict['splits_path'] = splits_path
    config_dict['results_path'] = results_path

    if osp.exists(results_path):
        with open(osp.join(results_path, 'config.yml'), 'r') as f:
            stored_config = yaml.safe_load(f)

        are_equals = True
        for k in config_dict:
            if k not in stored_config or stored_config[k] != config_dict[k]:
                are_equals = False

        if are_equals:
            eprint(f'Results dir {results_path} exists! We skip config with result!')
        else:
            raise FileExistsError('The results dir exists and its config file is different from the current one!')
    else:
        os.makedirs(results_path)

    with open(osp.join(results_path, 'config.yml'), 'w') as f:
        yaml.safe_dump(config_dict, f)

    return estimator, grid_dict, oversampling, one_hot


def run_model_selection(data_path, splits_path, results_path, config_filename, debug, n_jobs):

    estimator, grid_dict, *others = parse_config_file(data_path, splits_path, results_path, config_filename)

    hyperparams_names, hyperparams_values = get_grid(grid_dict)
    n_outer_splits, n_inner_splits = get_inner_outer_splits_size(splits_path)
    n_total_runs = n_outer_splits * n_inner_splits * reduce(mul, [len(x) for x in hyperparams_values])
    pbar = tqdm(desc='N runs', total=n_total_runs)

    def update_pbar(f=None):
        pbar.update(1)

    if not debug:
        # inizia il pool
        pool = ProcessPoolExecutor(max_workers=n_jobs)

    for id_config, c in enumerate(product(*hyperparams_values)):
        current_hyperparams_config = {hyperparams_names[i]: c[i] for i in range(len(c))}
        for i in range(n_outer_splits):
            for j in range(n_inner_splits):
                current_output_dir = osp.join(results_path, f'config_{id_config}', f'{i}_{j}')
                if not osp.exists(current_output_dir):
                    os.makedirs(current_output_dir)

                current_split_dir = osp.join(splits_path, f'{i}_{j}')
                if not debug:
                    f = pool.submit(run_single_config, data_path, current_split_dir, current_output_dir,
                            estimator, current_hyperparams_config, *others)
                    f.add_done_callback(update_pbar)
                else:
                    run_single_config(data_path, current_split_dir, current_output_dir,
                    estimator, current_hyperparams_config, *others)
                    update_pbar()

    # aspetto che tutti i processi finiscono
    if not debug:
        pool.shutdown(wait=True)

    print('Model selection ended!')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config-file', type=str, help='Configuration file to execute')
    parser.add_argument('--n-workers', default=60, type=int)
    parser.add_argument('--data-path', type=str, help='Path where data are stored')
    parser.add_argument('--splits-path', type=str, help='Path where splits are stored')
    parser.add_argument('--results-path', type=str, help='Path where results are saved')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_model_selection(args.data_path, args.splits_path, args.results_path, args.config_file, args.debug, args.n_workers)