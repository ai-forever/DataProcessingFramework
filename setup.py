import os
from setuptools import setup, find_packages


def get_requirements(filename: str = 'requirements.txt'):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires


setup(
    name="DPF",
    version="0.0.9",
    description="",
    author="Igor Pavlov, Mikhail Shoytov and Anastasia Lysenko",
    url='https://github.com/ai-forever/DataProcessingFramework',
    packages=find_packages(include=['DPF*']),
    install_requires=get_requirements(),
    extras_require={
        "filters": get_requirements(os.path.join('requirements', 'requirements_filters.txt')),
    }
)