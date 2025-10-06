# 🦟 Malaria Transmission Model – Madagascar

This repository contains a **deterministic malaria transmission model** that simulates the interaction between humans and mosquito vectors under varying **Insecticide-Treated Net (ITN)** coverage and resistance conditions.  

The model combines a **Susceptible–Infected–Recovered (SIR)** framework for the human population with a **Susceptible–Infected (SI)** framework for the mosquito population, implemented using **ordinary differential equations (ODEs)** solved in **R**.

---

## 📘 Model Overview

### Mathematical Framework
- **Humans:** \( S_h \rightarrow I_h \rightarrow R_h \)
- **Mosquitoes:** \( S_v \rightarrow I_v \)
- The model dynamically adjusts mosquito–human transmission rates based on **ITN efficacy decay**, reflecting the buildup of **insecticide resistance** over time.

### Key Equations

\[
E_t = E_{\text{initial}} \cdot e^{-\text{ResistanceRate} \cdot (t / 365)}
\]

\[
C_{\text{eff}} = \text{ITN}_{\text{Coverage}} \cdot E_t
\]

\[
T_{\text{red}} = 1 - C_{\text{eff}}
\]

Modified transmission rates:

\[
\beta_{v \rightarrow h} = \beta_{v,\text{base}} \cdot T_{\text{red}}, \quad
\beta_{h \rightarrow v} = \beta_{h,\text{base}} \cdot T_{\text{red}}
\]

For a complete explanation of the model equations, parameter definitions, and assumptions, see the accompanying article on **[Medium](https://medium.com/@freshsafoduker300/simulating-and-fitting-malaria-transmission-model-in-madagascar-impact-of-insecticide-treated-nets-fd9c10d4cda4)**.

---

## 🧩 Files

| File | Description |
|------|--------------|
| `Malaria_Transmission_Model_Madagascar.r` | Main R script implementing the SIR–SI model and simulations. |
| `Malaria-Transmission-Model-in-Madagascar.ipynb` | Jupyter notebook version (R kernel) for interactive exploration. |
| `outputs/` | Folder for generated plots, results, and sensitivity analyses. |

---

## ⚙️ Requirements

- **R version ≥ 4.0**
- Required packages:
  ```r
  install.packages(c("deSolve", "tidyverse"))
