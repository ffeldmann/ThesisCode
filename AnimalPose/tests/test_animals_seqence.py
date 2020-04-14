from AnimalPose.data.animals_sequence import AllAnimals_Sequence_Train, AllAnimals_Sequence_Validation
from tqdm import tqdm
config = {
    "animals": ["cat", "dog", "sheep", "horse"],
    "n_classes": 19,
    "n_channels": 3,  # rgb or grayscale
    "bilinear": True,
    "resize_to": 128,
    "as_grey": False,
    "sigma": 3,  # Good way sigma = size/64 for heatmaps
    "crop": True,
    "augmentation": False,
    "train_size": 0.8,
    "random_state": 42,
    "pck_alpha": 0.5,
    "pck_multi": False,
    "sequence_step_size": 1, # Steps to be taken from one frame to another
}

dtrain = AllAnimals_Sequence_Train(config)
dtest = AllAnimals_Sequence_Validation(config)


def test_get_example():
    # for element in dataset: print(element["frames"]().shape)
    ex = dtrain.get_example(5)

def test_fid_different():
    print()
    for element in tqdm(dtrain):
        assert element["fid0"] != element["fid1"]

# def test_train_test_split():
#     filenames = []
#     print()
#     for element in tqdm(dtrain):
#         filenames.append(element["framename0"])
#         filenames.append(element["framename1"])
#     print()
#     in_train_set = 0
#     for element in tqdm(dtest):
#         if element["framename0"] in filenames:
#             print(f"{element['framename0']} is in train set!")
#             in_train_set += 1
#         if element["framename1"] in filenames:
#             print(f"{element['framename1']} is in train set!")
#             in_train_set+=1
#     print(in_train_set)
#     assert in_train_set == 0

def test_all_shapes():
    size = config["resize_to"]
    print()
    for element in tqdm(dtrain):
        width, height, _ = element["inp0"].shape
        assert width == size and height == size, f"Width is {width}, height is {height}."
        width, height, _ = element["inp1"].shape
        assert width == size and height == size, f"Width is {width}, height is {height}."
    print()
    for element in tqdm(dtest):
        width, height, _ = element["inp0"].shape
        assert width == size and height == size, f"Width is {width}, height is {height}."
        width, height, _ = element["inp1"].shape
        assert width == size and height == size, f"Width is {width}, height is {height}."


def test_keypoints_in_range():
    size = config["resize_to"]
    print()
    for element in tqdm(dtrain):
        assert element["kps0"].any() >= 0
        assert element["kps0"].any() <= size
        assert element["kps1"].any() >= 0
        assert element["kps1"].any() <= size
    print()
    for element in tqdm(dtest):
        assert element["kps0"].any() >= 0
        assert element["kps0"].any() <= size
        assert element["kps1"].any() >= 0
        assert element["kps1"].any() <= size
