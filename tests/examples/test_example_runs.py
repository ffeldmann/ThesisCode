import os


def test_meta_dataset():

    """Just make sure example runs without errors."""
    args = [
        "-n",
        "meta",
        "-b",
        "AnimalPose/configs/MinimalNet.yaml",
        "AnimalPose/configs/Human36MFramesFlowMeta.yaml",
        "AnimalPose/configs/train.yaml",
        "tests/examples/config.yaml",
        "--num_steps",
        "11",
        "-t",
    ]
    return_value = os.system("edflow " + " ".join(args))
    assert return_value == 0, return_value


# def test_original_dataset():

#     """Just make sure example runs without errors."""
#     return_value = os.system(
#         "edflow -b AnimalPose/configs/MinimalNet.yaml AnimalPose/configs/Human36MFramesFlow.yaml AnimalPose/configs/train.yaml tests/examples/config.yaml -n meta --num_steps 11 --edeval_update_wandb_summary False -t"
#     )
#     assert return_value == 0, return_value


if __name__ == "__main__":
    test_origina_dataset()
    test_meta_dataset()
