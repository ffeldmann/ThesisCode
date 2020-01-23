import yaml
import argparse

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


def load_config(args=None):
    if isinstance(args, str):
        args = args.split(" ")

    opt, unknown = parser.parse_known_args(args=args)
    additional_kwargs = parse_unknown_args(unknown)

    config = dict()
    if opt.base:
        for base in opt.base:
            with open(base) as f:
                config.update(yaml.full_load(f))
    update_config(config, additional_kwargs)
    return config


if __name__ == "__main__":

    config = load_config()

    for k, v in sorted(config.items()):
        print("{}: {}".format(k, v))
