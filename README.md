![Alt text](Images/qsextra_logo.png?raw=true "Title")
# QSExTra: Quantum Simulation of Exciton Transport

## Installing the `qsextra` package and dependences
We advise the use of [Conda](https://www.anaconda.com/products/individual) for a clean setup.

Once Conda is installed, create a new environment
```
conda create --name qsextra_env python==3.11.5
```
and switch to it by running
```
conda activate qsextra_env
```

Clone the repository using
```
git clone https://github.com/federicogallina/qsextra.git
```
enter the folder and install the `qsextra` module with
```
pip install -e .
```

## Inside `qsextra`
The package is composed of two main modules:
- `qcomo` (quantum collision model): which can be used for dynamical simulations of exciton transport in chromophoric aggregates;
- `spectroscopy`: which can be used to simulate the optical response of an excitonic system. Simulation is achieved by implementing the excitation pathways described by double-sided Feynman diagrams as presented in [Bruschi, M.; Gallina, F.; Fresch, B. A Quantum Algorithm from Response Theory: Digital Quantum Simulation of Two-Dimensional Electronic Spectroscopy, J. Phys. Chem. Lett. 2024, 15, 5, 1484–1492](https://doi.org/10.1021/acs.jpclett.3c03499).
