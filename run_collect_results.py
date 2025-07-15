import os
os.environ['OMP_NUM_THREADS'] = '1'
import yaml
import numpy as np
from argparse import ArgumentParser
from itertools import product
from functools import reduce
from operator import mul
import os.path as osp
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score
from imblearn.metrics import sensitivity_score, specificity_score
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from utils import *


METRICS_y = {'B_ACC': balanced_accuracy_score, 'ACC': accuracy_score,
             'SPEC': lambda *xx: specificity_score(*xx, average='macro'),
             'SENS': lambda *xx: sensitivity_score(*xx, average='macro'),
             'RECALL': lambda *xx: recall_score(*xx, average='macro'),
             'PRECISION': lambda *xx: precision_score(*xx, average='macro'),
             'F1': lambda *xx: f1_score(*xx, average='macro')}
METRICS_p = {'ROC_AUC': lambda *xx: roc_auc_score(*xx, multi_class='ovo', average='macro')}


def compute_metrics(y_true, y_pred, p_pred):
    metrics_d = {}
    for t in y_true:
        for k, m in METRICS_y.items():
            metrics_d[f'{t}_{k}'] = m(y_true[t], y_pred[t])
        if p_pred is not None:
            for k, m in METRICS_p.items():
                metrics_d[f'{t}_{k}'] = m(y_true[t], p_pred[t]) if p_pred[t] is not None else -1

    return metrics_d

def compute_model_selection_metrics(id_config, data_path, splits_path, output_path):
    n_outer_splits, n_inner_splits = get_inner_outer_splits_size(splits_path)

    # LOAD the data
    y = np.load(osp.join(data_path, 'y.npy'), allow_pickle=True)

    all_out_d = {}
    for i in range(n_outer_splits):
        for j in range(n_inner_splits):
            # carico i risultati
            current_output_dir = osp.join(output_path, f'config_{id_config}', f'{i}_{j}')
            current_split_dir = osp.join(splits_path, f'{i}_{j}')
            results_file = osp.join(current_output_dir, 'results.npz')

            if not osp.exists(results_file):
                eprint(f'Config {id_config} is not completed!')
                return None

            results_dict = np.load(results_file)

            # carico gli split
            train_idx, val_idx, test_idx = get_split_idx(current_split_dir)

            y_true = {'tr': y[train_idx], 'val': y[val_idx], 'test': y[test_idx]}
            y_pred = {'tr': results_dict['ypred_tr'], 'val': results_dict['ypred_val'], 'test': results_dict['ypred_test']}
            p_pred = None
            if 'proba_tr' in results_dict:
                p_pred = {'tr': results_dict['proba_tr'], 'val': results_dict['proba_val'], 'test':  results_dict['proba_test']}

            metrics_d = compute_metrics(y_true, y_pred, p_pred)
            metrics_d['elapsed_time'] = results_dict['elapsed_time']

            all_out_d[(id_config, i, j)] = metrics_d

    return all_out_d


def get_model_selection_exp_results(results_path):
    config_filename = osp.join(results_path, 'config.yml')
    with open(config_filename, 'r') as f:
        config_dict = yaml.safe_load(f)

    data_path = config_dict['data_path']
    splits_path = config_dict['splits_path']

    if osp.exists(osp.join(results_path, 'all_results.npz')):
        eprint(f'Results file already exists! WE LOAD IT!')
        all_results = np.load(osp.join(results_path, 'all_results.npz'), allow_pickle=True)['all_metrics'].item()
    else:
        n_outer_splits, n_inner_splits = get_inner_outer_splits_size(splits_path)
        hyperparams_names, hyperparams_values = get_grid(config_dict['hyperparams_grid'])
        n_configs = reduce(mul, [len(x) for x in hyperparams_values])
        n_total_runs = n_outer_splits * n_inner_splits * n_configs

        pool = ProcessPoolExecutor(max_workers=20)
        pbar = tqdm(desc='N runs', total=n_total_runs)
        all_results = {}

        def _collect_result(f):
            pbar.update(n_inner_splits*n_outer_splits)
            out = f.result()
            if out is not None:
                for (id_config, i, j), res_d in out.items():
                    for k, v in res_d.items():
                        if k not in all_results:
                            all_results[k] = np.empty((n_outer_splits, n_inner_splits, n_configs))
                        all_results[k][i, j, id_config] = v

        c_list = []
        for id_config, c in enumerate(product(*hyperparams_values)):
            c_list.append(c)
            f = pool.submit(compute_model_selection_metrics, id_config, data_path, splits_path, results_path)
            f.add_done_callback(_collect_result)

        pool.shutdown(wait=True)

        all_res_file = osp.join(results_path, 'all_results')
        np.savez(all_res_file, all_metrics=all_results, hyperparams_names=hyperparams_names, config_list=c_list)

    best_config_idx, best_results = get_best_val_results(all_results, 'B_ACC')
    return best_results, best_config_idx, all_results


def get_test_exp_results(results_path):
    config_filename = osp.join(results_path, 'config.yml')
    with open(config_filename, 'r') as f:
        config_dict = yaml.safe_load(f)

    data_path = config_dict['data_path']
    splits_path = config_dict['splits_path']
    output_path = osp.join(config_dict['results_path'], 'test')

    if osp.exists(osp.join(output_path, 'test_results.npz')):
        eprint(f'Results file already exists! WE LOAD IT!')
        test_results = np.load(osp.join(results_path, 'test_results.npz'), allow_pickle=True)
    else:
        n_outer_splits, n_inner_splits = get_inner_outer_splits_size(splits_path)
        test_results = {}
        y = np.load(osp.join(data_path, 'y.npy'), allow_pickle=True)
        for i in tqdm(range(n_outer_splits)):
            current_split_dir = osp.join(splits_path, f'{i}_0')
            results_file = osp.join(output_path, f'{i}', 'results.npz')
            results_dict = np.load(results_file)

            # carico gli split
            train_idx, val_idx, test_idx = get_split_idx(current_split_dir)
            train_idx = train_idx + val_idx
            val_idx = None

            y_true = {'tr': y[train_idx], 'test': y[test_idx]}
            y_pred = {'tr': results_dict['ypred_tr'], 'test': results_dict['ypred_test']}
            p_pred = None
            if 'proba_tr' in results_dict:
                p_pred = {'tr': results_dict['proba_tr'], 'test': results_dict['proba_test']}

            metrics_d = compute_metrics(y_true, y_pred, p_pred)
            metrics_d['elapsed_time'] = results_dict['elapsed_time']

            for k, v in metrics_d.items():
                if k not in test_results:
                    test_results[k] = np.empty((n_outer_splits,))
                test_results[k][i] = v

        test_res_file = osp.join(results_path, 'test_results')
        np.savez(test_res_file, test_metrics=test_results)

    return test_results


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('results_path')
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.test:
        test_results = get_test_exp_results(args.results_path)
        for k, v in test_results.items():
            if 'ACC' in k:
                v = 100*v
            print(f'{k}: {v.mean(axis=0):0.2f} +/- {v.std(axis=0):0.2f}')
    else:
        best_results, best_config_idx, all_results = get_model_selection_exp_results(args.results_path)
        for k, v in best_results.items():
            if 'ACC' in k:
                v = 100*v
            print(f'{k}: {v.mean(axis=0):0.2f} +/- {v.std(axis=0):0.2f}')