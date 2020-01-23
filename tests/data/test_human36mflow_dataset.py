import os
import sys
import numpy as np
import time
import tqdm
import socket
from edflow.util import pp2mkdtable
from edflow.iterators.batches import make_batches
from edflow.main import get_impl

sys.path.append(os.path.dirname("."))

from AnimalPose.scripts.load_config import load_config


def get_config():
    config = {
        "hostname": socket.gethostname(),
        "append_times": False,
        "batch_size": 2,
        "n_data_processes": 1,
        "split_type": "action",
    }
    config_ = load_config("-b AnimalPose/configs/Human36MFramesFlowMeta.yaml")
    config.update(config_)

    return config


def test_load_examples(config=None):
    if config is None:
        config = get_config()
    times = {"create": {}, "load": {}}

    # create dataset
    before = time.time()
    print(config)
    dataset = get_impl(config, "dataset")(config)
    print("Created dataset {}".format(type(dataset)))
    times["create"]["dataset"] = time.time() - before

    # load single examples
    if config["append_times"]:
        import wandb
    before = time.time()
    for i in tqdm.tqdm(range(100), desc="dataset examples"):
        ex = dataset[i]
        if config["append_times"]:
            wandb.log({"times_ex": ex["times"]}, step=i)
    times["load"]["examples"] = time.time() - before

    # create batches
    try:
        before = time.time()
        batches = make_batches(
            dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            n_processes=config["n_data_processes"],
            n_prefetch=1,
        )
        times["create"]["batches"] = time.time() - before

        # load batches
        before = time.time()
        for i in tqdm.tqdm(range(100), desc="batches"):
            next(batches)
        times["load"]["batches"] = time.time() - before
    except Exception as error:
        assert False, error
    finally:
        batches.finalize()

    return times


def execute_benchmark(**kwargs):
    import wandb
    import socket
    import datetime

    config = get_config()
    config.update(**kwargs)
    print("config", config)

    wandb_project = "benchmark_dataset_loading"
    run_name = "{}_{}".format(socket.gethostname(), datetime.datetime.now().isoformat())
    run = wandb.init(project=wandb_project, name=run_name, config=config, reinit=True,)

    times = test_load_examples(config=config)
    print("times", times)
    iterations_per_s = {k: N_ITERATIONS / v for k, v in times["load"].items()}
    print("iterations per second", iterations_per_s)
    examples_per_s = {
        "examples": iterations_per_s["examples"],
        "batches": iterations_per_s["batches"] * config["batch_size"],
    }
    print("examples per second", examples_per_s)

    wandb.log(
        {
            "times": times,
            "iterations_per_s": iterations_per_s,
            "examples_per_s": examples_per_s,
        }
    )
    api = wandb.Api()
    api_run = api.run("hperrot/{}/{}".format(wandb_project, run.id))
    means = api_run.history().filter(regex="times_ex").mean().to_dict()
    means = {
        k: v for k, v in sorted(means.items(), key=lambda item: item[1], reverse=True)
    }
    means_data = [[k, v] for k, v in means.items()]
    means_table = wandb.Table(columns=["Process", "Mean time [s]"], data=means_data)
    wandb.log({"times_table": means_table})
    print(pp2mkdtable(means, jupyter_style=True))


def test_Human36MFramesFlowMeta_FlowPrep(config=None):
    if config is None:
        config = get_config()

    from AnimalPose.data import Human36MFramesFlowMeta_FlowPrep

    dataset_flowprep = Human36MFramesFlowMeta_FlowPrep(config)

    ex_i = dataset_flowprep[0]
    for i in tqdm.tqdm(range(100), desc="Human36MFramesFlowMeta_FlowPrep"):
        ex_i = dataset_flowprep[i]


def test_Human36MFramesMeta(config=None):
    if config is None:
        config = get_config()

    from AnimalPose.data.human36m_meta import Human36MFramesMeta

    dataset_images = Human36MFramesMeta(config)

    ex_i = dataset_images[0]
    for i in tqdm.tqdm(range(100), desc="Human36MFramesMeta"):
        ex_i = dataset_images[i]


def has_stuff(ex):
    assert type(ex["forward_flow"]) == np.ndarray, type(ex["forward_flow"])
    assert type(ex["images"][0]["image"]) == np.ndarray, type(ex["forward_flow"])
    assert type(ex["backward_flow"]) == np.ndarray, type(ex["forward_flow"])


def test_Human36MFramesFlowMeta_Train(config=None):
    if config is None:
        config = get_config()

    from AnimalPose.data import Human36MFramesFlowMeta_Train

    dataset_flow = Human36MFramesFlowMeta_Train(config)

    ex_i = dataset_flow[0]
    has_stuff(ex_i)
    for i in tqdm.tqdm(range(100), desc="Human36MFramesFlowMeta_Train"):
        ex_i = dataset_flow[i]


def test_Human36MFramesFlow(config=None):
    if config is None:
        config = get_config()

    from AnimalPose.data.human36m_dataset import Human36MFramesFlow

    dataset_flow = Human36MFramesFlow(config)

    ex_i = dataset_flow[0]
    has_stuff(ex_i)
    for i in tqdm.tqdm(range(100), desc="Human36MFramesFlow"):
        ex_i = dataset_flow[i]


if __name__ == "__main__":

    # execute_benchmark()
    test_load_examples()
    test_Human36MFramesFlow()
    test_Human36MFramesMeta()
    test_Human36MFramesFlowMeta_Train()
    test_Human36MFramesFlowMeta_FlowPrep()
