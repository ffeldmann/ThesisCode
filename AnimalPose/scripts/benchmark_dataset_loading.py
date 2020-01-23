import os
import sys
import argparse
import yaml

sys.path.append(os.path.dirname("."))
# sys.path.append(os.getcwd())

from tests.data.test_human36mflow_dataset import execute_benchmark

from edflow.config import parse_unknown_args, update_config


parser = argparse.ArgumentParser()

parser.add_argument(
    "-b",
    "--base",
    nargs="*",
    metavar="base_config.yaml",
    help="paths to base configs. Loaded from left-to-right. "
    "Parameters can be overwritten or added with command-line options of the form `--key value`.",
    default=None,
)


def load_config(base_configs, additional_kwargs):
    config = dict()
    if base_configs:
        for base in base_configs:
            with open(base) as f:
                config.update(yaml.full_load(f))
    update_config(config, additional_kwargs)
    return config


if __name__ == "__main__":

    opt, unknown = parser.parse_known_args()
    additional_kwargs = parse_unknown_args(unknown)

    config = load_config(opt.base, additional_kwargs)
    for batch_size in [64, 32, 16, 8, 4, 2, 1]:
        for n_data_processes in [16, 8, 4, 2, 1]:
            if n_data_processes > batch_size:
                continue
            config.update(batch_size=batch_size, n_data_processes=n_data_processes)
            execute_benchmark(**config)
