from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    Reads requirements.txt and returns a list of dependencies.
    Removes '-e .' if present.
    """
    requirements = []

    with open(file_path, encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f.readlines()
            if line.strip() and line.strip() != HYPHEN_E_DOT
        ]

    return requirements


setup(
    name="mlproject",
    version="0.0.1",
    author="neetcoder29",
    author_email="valoroustruth@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
