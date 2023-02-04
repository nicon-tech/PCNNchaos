# PCNNchaos
Piecewise integrable neural network: an interpretable chaos identification framework

# Abstract
Artificial neural networks (ANNs) are an effective data-driven approach to model chaotic dynamics. Although ANNs are universal approximators that easily incorporate mathematical structure, physical information, and constraints, they are scarcely interpretable. Here, we develop a neural network framework in which the chaotic dynamics is reframed into piecewise models. The discontinuous formulation defines switching laws representative of the bifurcations mechanisms, recovering the system of differential equations and its primitive (or integral), which describe the chaotic regime.
Interpretation of chaotic dynamics from data is an important task in engineering,1 meteorology,2 and other fields of science. Among various aspects of measure and description of chaotic dynamics, much interest is posed in the integrability,3 i.e., “…
integrable systems …
a formula could be found for all time describing a system’s future state.” as stated by Moore.4 Studying and proving the existence of a chaotic attractor from a data-driven perspective is a demanding regression problem in which machine learning (ML) techniques offer a valuable assistance. The synthesis of a piecewise-smooth modeling through neural networks gives a valid and rigorous reconstruction of the dynamical structure and bifurcations with the option of providing an integrable representation. In this work, a minimal mathematically biased artificial neural network is introduced to extract analytically tractable piecewise-smooth dynamical systems that directly account for the integrability condition.

# Code
The repository is composed of two folders containing the implementation of the algorithm for the two numerical examples presented in the associated paper ("Lorenz system" and "kuramoto-sivashinsky equation"). In order to show the incredible generality of the approach the two cases are solved by applying two different implementations, which are the DMD for the "kuramoto-sivashinsky equation" and the Deep Koopman Neural Net for the "Lorenz system". Each folder contain its specific readme file.
