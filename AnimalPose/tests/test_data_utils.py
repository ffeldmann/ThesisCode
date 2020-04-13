from AnimalPose.data.util import make_heatmaps, gaussian_k
import skimage.io
import numpy as np

keypoints = np.array(
    [[5, 5],
     [212, 217],
     [187, 216],
     [200, 232],
     [223, 194],
     [180, 190],
     [146, 246],
     [222, 263],
     [199, 246],
     [161, 259],
     [1, 1],
     ])
# Image tests/examples/example_cat.jpg

def test_make_heatmaps():
    image = skimage.io.imread("tests/examples/example_cat.jpg", as_gray=True)
    heatmaps = make_heatmaps(image, keypoints)
    assert image.shape[0] == heatmaps.shape[1]
    assert image.shape[1] == heatmaps.shape[2]
    for idx, kpt in enumerate(keypoints):
        # check if the middle of the keypoint is white
        assert heatmaps[idx][int(kpt[1]), int(kpt[0])] == 1
        # check if other points are zero
        assert heatmaps[idx][int(image.shape[0])-1, int(image.shape[1])-1] == 0

def test_gaussian_k():
    pass


if __name__ == "__main__":
    test_make_heatmaps()