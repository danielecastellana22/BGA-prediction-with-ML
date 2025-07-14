import os.path as osp
import sys
from pydoc import locate
import numpy as np


__all__ =['get_split_idx', 'get_inner_outer_splits_size', 'string2class', 'eprint', 'get_grid',
          'get_best_val_results']


def get_split_idx(current_split_dir):
    with open(osp.join(current_split_dir, 'train_idx.txt'), 'r') as f:
        train_idx = [int(l.strip()) for l in f.readlines()]
    with open(osp.join(current_split_dir, 'val_idx.txt'), 'r') as f:
        val_idx = [int(l.strip()) for l in f.readlines()]
    with open(osp.join(current_split_dir, 'test_idx.txt'), 'r') as f:
        test_idx = [int(l.strip()) for l in f.readlines()]

    return train_idx, val_idx, test_idx


def get_inner_outer_splits_size(splits_path):
    return map(int, osp.basename(splits_path).split('_'))


def string2class(string):
    c = locate(string)
    if c is None:
        raise ModuleNotFoundError('{} cannot be found!'.format(string))
    return c


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def get_grid(grid_dict):
    hyperparams_names, hyperparams_values = zip(*grid_dict.items())
    hyperparams_values = [x if isinstance(x, list) else [x] for x in hyperparams_values]
    return hyperparams_names, hyperparams_values


def get_best_val_results(all_metrics_dict, m_name_to_maximise):
    mk_name = f'val_{m_name_to_maximise}'
    n_outer_splits = all_metrics_dict[mk_name].shape[0]

    best_config_idx = all_metrics_dict[mk_name].mean(axis=1).argmax(axis=1)

    best_results = {}
    for i in range(n_outer_splits):
        for k, v in all_metrics_dict.items():
            if k not in best_results:
                best_results[k] = np.empty(n_outer_splits)
            best_results[k][i] = v[i, :, best_config_idx[i]].mean()

    return best_config_idx, best_results