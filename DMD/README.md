# The present algorithm leverages on the work of Kratsios et al.
[Learning Sub-Patterns in Piece-Wise Continuous Functions](https://arxiv.org/pdf/2010.15571.pdf)
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

# Main code
The main codes are Architope_Chaos_DMD.ipynb and NewArchitope_Chaos_DMD.ipynb

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

2. Preprocessing data, train models and obtaining predictions can all be done by executing the following codes:
Architope_Chaos_DMD.ipynb
NewArchitope_Chaos_DMD.ipynb

The first (Architope_Chaos_DMD.ipynb) trains the model simply by training the N sub-models that build the overall neural network. The second (NewArchitope_Chaos_DMD.ipynb) loop the process of learning sub-models and learning the deep classifier inproving the interaction between these two phases.

---
