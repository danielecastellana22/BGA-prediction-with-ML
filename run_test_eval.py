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
    train_idx = train_idx + val_idx
    val_idx = None

    # LOAD the data
    X = np.load(osp.join(data_path, 'X.npy'))
    y = np.load(osp.join(data_path, 'y.npy'))
    X_tr, y_tr = X[train_idx], y[train_idx]
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

    # Calculate the execution time of the fit
    start_time = time.time()
    m.fit(X_tr_os, y_tr_os)
    end_time = time.time()
    elapsed_time = end_time - start_time

    X_test = preprocessing_pipeline.transform(X_test)

    ypred_tr = m.predict(X_tr)
    ypred_test = m.predict(X_test)

    try:
        proba_tr = m.predict_proba(X_tr)
        proba_test = m.predict_proba(X_test)
    except Exception:
        proba_tr = proba_test = None

    results_dict = {'elapsed_time': elapsed_time,
                    'ypred_tr': ypred_tr, 'ypred_test': ypred_test}

    if proba_tr is not None:
        results_dict.update({'proba_tr': proba_tr, 'proba_test': proba_test})

    np.savez(results_file, **results_dict)


def parse_config_file(results_path):
    config_filename = osp.join(results_path, 'config.yml')

    with open(config_filename, 'r') as f:
        config_dict = yaml.safe_load(f)

    if not osp.exists(osp.join(results_path, 'all_results.npz')):
        raise FileNotFoundError(f'all_results.npz not found in {results_path}')

    all_results = np.load(osp.join(results_path, 'all_results.npz'), allow_pickle=True)

    data_path = config_dict['data_path']
    splits_path = config_dict['splits_path']

    estimator = string2class(config_dict['estimator'])
    oversampling = config_dict['oversampling']
    one_hot = config_dict['one_hot']
    grid_dict = config_dict['hyperparams_grid']

    return all_results, data_path, splits_path, estimator, grid_dict, oversampling, one_hot


def run_test_eval(results_path, debug, n_jobs):

    all_results, data_path, splits_path, estimator, grid_dict, *others = parse_config_file(results_path)

    hyperparams_names, hyperparams_values = get_grid(grid_dict)
    n_outer_splits, n_inner_splits = get_inner_outer_splits_size(splits_path)
    n_total_runs = n_outer_splits
    pbar = tqdm(desc='N runs', total=n_total_runs)

    def update_pbar(f=None):
        pbar.update(1)

    if not debug:
        # inizia il pool
        pool = ProcessPoolExecutor(max_workers=n_jobs)

    best_config_idx, _ = get_best_val_results(all_results['all_metrics'].item(), 'B_ACC')
    all_configs = list(product(*hyperparams_values))

    for i in range(n_outer_splits):
        id_config = best_config_idx[i]
        c = all_configs[id_config]
        current_output_dir = osp.join(results_path, f'test', f'{i}')

        if not osp.exists(current_output_dir):
            os.makedirs(current_output_dir)

        current_split_dir = osp.join(splits_path, f'{i}_0')
        current_hyperparams_config = {hyperparams_names[k]: c[k] for k in range(len(c))}

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

    print('Model assessment ended!')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--n-workers', default=10, type=int)
    parser.add_argument('--results-path', type=str, help='Path where results are saved')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_test_eval(args.results_path, args.debug, args.n_workers)