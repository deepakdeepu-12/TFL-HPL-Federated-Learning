from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith('#')]

setup(
    name="tfl-hpl",
    version="1.0.0",
    author="Burra Deepak Yadav",
    author_email="deepakyadavdeepu94@gmail.com",
    description="Trustworthy Federated Learning with Heterogeneous Privacy Levels",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deepakdeepu-12/TFL-HPL-Federated-Learning",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "tfl-hpl=tfl_hpl.cli:main",
        ],
    },
)
