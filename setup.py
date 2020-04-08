from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="AnimalPose",
    version="0.0.1",
    author="Hannes Perrot",
    author_email="hp96@gmx.de",
    description="Next video frame generation with optical flow prior",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ffeldman/AnimalPose",
    include_package_data=True,
    packages=find_packages(),
    install_requires=["numpy", "tqdm", "matplotlib", "Pillow",],
    classifiers=["Programming Language :: Python :: 3",],
)
