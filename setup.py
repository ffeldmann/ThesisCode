from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="AnimalPose",
    version="1.0.0",
    author="Felix Feldmann",
    author_email="felix@bnbit.de",
    description="Animal Pose Estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ffeldman/AnimalPose",
    include_package_data=True,
    packages=find_packages(),
    install_requires=["numpy", "tqdm", "matplotlib", "Pillow",],
    classifiers=["Programming Language :: Python :: 3",],
)
