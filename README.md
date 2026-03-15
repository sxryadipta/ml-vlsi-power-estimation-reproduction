# ML-Based Power Estimation for CMOS VLSI Circuits (Paper Reproduction)

This repository reproduces results from:

[Machine Learning Based Power Estimation for CMOS VLSI Circuits](https://www.researchgate.net/publication/353749810_Machine_Learning_Based_Power_Estimation_for_CMOS_VLSI_Circuits)

Models implemented:
- Backpropagation Neural Network
- Random Forest

Dataset:
ISCAS'89 benchmark circuits.

Features:
Gate count
AND gates
Inverters
NOR
NAND
OR
DFF
Inputs
Outputs

Target:
Power (mW)

Metrics:
MSE
RMSE
R²
Prediction Error %

Expected Result:
Random Forest performs better than BPNN with lower MSE.