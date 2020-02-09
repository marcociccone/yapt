import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="yapt",
    version="0.0.1",
    author="Marco Ciccone",
    author_email="marco.ciccone@polimi.it",
    description="Yet Another PyTorch Trainer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/mciccone/yapt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
