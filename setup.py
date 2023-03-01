from setuptools import setup, find_packages

setup(
    name='qsextra',
    version=0.3,
    author='Federico Gallina',
    author_email='federico.gallina@unipd.it',
    description='Quantum Simulation of Exciton Transport',
    long_description='Simulate exciton transport in molecular network in  presence of a dephasing environment. It is based on circuit quantum computing using the IBM Qiskit suite.',
    packages=find_packages(),
    url='https://github.com/federicogallina/qsextra',
    install_requires=['numpy','qiskit']
)