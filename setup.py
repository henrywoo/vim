from setuptools import setup
import os

here = os.path.dirname(os.path.realpath(__file__))
HAS_CUDA = os.system("nvidia-smi > /dev/null 2>&1") == 0

VERSION = "0.0.1.dev1"
DESCRIPTION = "VITVQGAN - VECTOR-QUANTIZED IMAGE MODELING WITH IMPROVED VQGAN"

packages = [
    "vitvqgan",
]


def read_file(filename: str):
    try:
        lines = []
        with open(filename) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines if not line.startswith("#")]
        return lines
    except:
        return []


def package_files(ds):
    paths = []
    for d in ds:
        for path, directories, filenames in os.walk(d):
            for filename in filenames:
                t = str(os.path.join(path, filename))
                if "__pycache__" not in t:
                    paths.append(t[len("src/vitvqgan/") :])
    return paths


extra_files = package_files(["src/vitvqgan/"])

setup(
    name="vitvqgan",
    version=VERSION,
    author="Fuheng Wu",
    description=DESCRIPTION,
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    install_requires=read_file(f"{here}/requirements.txt"),
    keywords=[
        "vitvqgan",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    packages=packages,
    include_package_data=True,
    package_dir={"": "src"},
    package_data={"vitvqgan": extra_files},
    url="https://github.com/henrywoo/vim",
)
