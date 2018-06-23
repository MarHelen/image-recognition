# setup.py

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="image-recognition-tool",
    version="0.0.1",
    author="Olena Marushchenko",
    author_email="markohelen@gmail.com",
    description="Landscape images recognition tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MarHelen/image-recognition-tool",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
