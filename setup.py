from setuptools import setup, find_packages

setup(
    name='qsextra',
    version=0.1,
    author='Federico Gallina',
    author_email='federico.gallina@unipd.it',
    description='Quantum Simulation of Exciton Transport',
    long_description='Simulate exciton transport in molecular networks and their time-resolved electronic spectra.',
    packages=find_packages(),
    url='https://github.com/federicogallina/qsextra',
    install_requires=['numpy==1.26.4',
                      'scipy==1.12',
                      'matplotlib',
                      'qiskit==1.0.0',
                      'qiskit-aer==0.13.3',
                      'qutip>=4.7.5',
                     ]
)
