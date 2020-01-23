import os
import argparse
import glob
import tqdm
from PIL import Image


parser = argparse.ArgumentParser(
    description="Samples down images in Human3.6M dataset to given square resolution"
)
parser.add_argument(
    "--dataroot",
    type=str,
    default="/export/scratch/hperrot/Datasets/human3.6M",
    help="root of the dataset",
)
parser.add_argument("--load_size", type=int, default=256, help="size to downsample")
parser.add_argument("--file_extension", type=str, default="jpg")


def downsample(dataroot, load_size, file_extension):
    if not "/processed/all/" in dataroot:
        dataroot = os.path.join(dataroot, "processed/all")
    search_pattern = os.path.join(dataroot, "**", "*." + file_extension)
    print("search_pattern", search_pattern)
    for old_filepath in tqdm.tqdm(glob.iglob(search_pattern, recursive=True)):
        new_filepath = old_filepath.replace(
            "/processed/all/", "/downsampled/{}/".format(load_size)
        )
        # print(old_filepath)
        # print(new_filepath)

        if not os.path.isfile(new_filepath):
            os.makedirs(os.path.dirname(new_filepath), exist_ok=True)

            image = Image.open(old_filepath)
            image = image.resize((load_size, load_size), Image.BILINEAR)
            image.save(new_filepath, quality=95)


if __name__ == "__main__":

    args = parser.parse_args()

    print(args)

    downsample(**vars(args))
