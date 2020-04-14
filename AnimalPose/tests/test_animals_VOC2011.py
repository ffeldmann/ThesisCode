from AnimalPose.data.animals_VOC2011 import AllAnimalsVOC2011_Train, AllAnimalsVOC2011_Validation
from tqdm import tqdm
config = {
    "animals": ["cat", "dog", "sheep", "horse", "cow"],
    "n_classes": 20,
    "n_channels": 3,  # rgb or grayscale
    "bilinear": True,
    "resize_to": 128,
    "as_grey": False,
    "sigma": 2,  # Good way sigma = size/64 for heatmaps
    "crop": True,
    "augmentation": True,
    "train_size": 0.8,
    "random_state": 42,
    "pck_alpha": 0.5,
    "pck_multi": False,
}

dtrain = AllAnimalsVOC2011_Train(config)
dtest = AllAnimalsVOC2011_Validation(config)


def test_get_example():
    # for element in dataset: print(element["frames"]().shape)
    ex = dtrain.get_example(5)

def test_all_shapes():
    size = config["resize_to"]
    print()
    for element in tqdm(dtrain):
        width, height, _ = element["inp0"].shape
        assert width == size and height == size, f"Width is {width}, height is {height}."
    print()
    for element in tqdm(dtest):
        width, height, _ = element["inp0"].shape
        assert width == size and height == size, f"Width is {width}, height is {height}."


def test_keypoints_in_range():
    size = config["resize_to"]
    print()
    for element in tqdm(dtrain):
        assert element["kps"].any() >= 0
        assert element["kps"].any() <= size
    print()
    for element in tqdm(dtest):
        assert element["kps"].any() >= 0
        assert element["kps"].any() <= size
