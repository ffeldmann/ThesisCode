from AnimalPose.data.util import make_heatmaps, gaussian_k
import skimage
kpts = np.array(
        [[212.27 , 217.87 ],
        [187.5  , 216.06 ],
        [200.62 , 232.   ],
        [223.27 , 194.05 ],
        [180.77 , 190.11 ],
        [146.64 , 246.13 ],
        [222.77 , 263.87 ],
        [  0.   ,   0.   ],
        [199.97 , 246.13 ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ],
        [161.875, 259.375],
        [  0.   ,   0.   ],
        [  0.   ,   0.   ]])
# Image tests/examples/example_cat.jpg

def test_make_heatmaps():
    image = skimage.io.imread("tests/examples/example_cat.jpg", as_gray=True)
    pass

def test_gaussian_k():
    pass