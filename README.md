![Alt text](images/qsextra_logo_V2.png?raw=true "Title")
# QSExTra
Quantum Simulation of Exciton Transport

## Installing the `qsextra` package and dependences
We advise the use of [Conda](https://www.anaconda.com/products/individual) for a clean setup.

The software requires the following packages to be installed:

- [python3](https://www.python.org/)
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [matplotlib](https://matplotlib.org/)
- [qiskit](https://qiskit.org/)
- [qiskit-aer](https://qiskit.org/)
- [qutip](https://qutip.org/)

Once Conda is installed, create a new environment
```
conda create --name qsextra_env python>=3.11.5
```
and switch to it by running
```
conda activate qsextra_env
```

Clone the repository using
```
git clone https://github.com/federicogallina/qsextra.git
```

From the parent repository folder, in your `qsextra_env`, the requirements can be installed with
```
pip install -r requirements
```

Then, run the following to install the `qsextra` package
```
pip install -e .
```
