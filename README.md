# 🦟 Malaria Transmission Model in Madagascar

This repository contains the R-based simulation and fitting of a **malaria transmission model** that combines an SIR (Susceptible–Infected–Recovered) framework for the human population and an SI (Susceptible–Infected) framework for the mosquito population. 

The model incorporates **Insecticide-Treated Net (ITN)** effectiveness that decays over time to simulate growing insecticide resistance — a key public health challenge in malaria control.

## 📂 Contents
- `Malaria-Transmission-Model-in-Madagascar.R`: R script implementing the model equations.
- `Malaria-Transmission-Model-in-Madagascar.R.ipynb`: Notebook version of the simulation.
- `figures/`: Model output plots and figures.

## ⚙️ Requirements
- R version ≥ 4.0  
- Packages: `deSolve`, `tidyverse`

## 🚀 How to Run
```R
# Run the model simulation
source("Malaria-Transmission-Model-in-Madagascar.R")
