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
This package has been used to obtain results presented in [Gallina, F.; Bruschi, M.; Fresch, B. Simulating Non-Markovian Dynamics in Multidimensional Electronic Spectroscopy via Quantum Algorithm (2024)](https://doi.org/10.48550/arXiv.2409.05548).

The package is composed of two main modules:
- `qcomo` (quantum collision model): Simulation of the exciton dynamics in chromophore aggregates. The open quantum system dynamics is implemented in terms of quanutm circuits (`qevolve`) using a collision model. Pseudomode technique can be used to account for a non-Markovian environment. A classical execution of the system-pseudomode (`clevolve`) dynamics is also available, based on Qutip master equation solver;
- `spectroscopy`: Simulation of the optical response of an excitonic system. Simulation is achieved by implementing the excitation pathways described by double-sided Feynman diagrams as presented in [Bruschi, M.; Gallina, F.; Fresch, B. A Quantum Algorithm from Response Theory: Digital Quantum Simulation of Two-Dimensional Electronic Spectroscopy, J. Phys. Chem. Lett. 2024, 15, 5, 1484–1492](https://doi.org/10.1021/acs.jpclett.3c03499).
