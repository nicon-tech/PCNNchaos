# The present algorithm leverages on the work of Kratsios et al.
# [Learning Sub-Patterns in Piece-Wise Continuous Functions](https://arxiv.org/pdf/2010.15571.pdf)
**Submitted to:** *Thirty-eighth International Conference on Machine Learning*

---
#### Cite As:
        @misc{ZamanlooyKratsios2021PiecewiseContinuousPCNNs,
                title={Learning Sub-Patterns in Piecewise Continuous Functions}, 
                author={Anastasis Kratsios and Behnoosh Zamanlooy},
                year={2021},
                eprint={2010.15571},
                archivePrefix={arXiv},
                primaryClass={cs.NE}
               }

---

## Requirements

To install requirements:
*  Install [Anaconda](https://www.anaconda.com/products/individual)  version 4.8.2.
* Create Conda Environment
``` pyhton
# cd into the same directory as this README file.

conda create python=3.8 --name architopes \
conda activate architopes \
pip install -r requirements.txt
```
---

## Organization of directory:
 - Data in the "inputs" sub-directory,
 - All model outputs go to the "outputs" subdirectory,
 - Jupyternotebook versions of the python files and auxiliary "standalone python scripts" are found in the "Auxiliary_Codes" subdirectory.  

---

## Preprocessing, Training, and Evaluating
1. Specify the parameters related to each set and the space of hyper parameters in the "Grid_Enhanced_Network.py" script.   

2. Preprocessing data, train models and obtaining predictions can all be done by executing the following commands:
python3.8 Architope.py
python3.8 Architope_Expert.py
python3.8 Architope_Expert_Semi_Supervised_Standalone.py

The first trains the semi-supervised architope model and the following benchmark modelds: (Vanilla) ffNN, Grad.Bstd Rand.F, Bagged Architope, and Architope-logistic.  The subsequent two scripts, respectively, train the architope on the expert-provided partition described [here](https://github.com/ageron/handson-ml/tree/master/datasets/housing) and architope obtained from an additional repartitioning step, as described by Algorithm 3.2 of the paper.  

---
