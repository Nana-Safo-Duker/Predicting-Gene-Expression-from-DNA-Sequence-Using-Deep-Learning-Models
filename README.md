# 🦟 Malaria Transmission Model – Madagascar

This repository contains a **deterministic malaria transmission model** that simulates the interaction between humans and mosquito vectors under varying **Insecticide-Treated Net (ITN)** coverage and resistance conditions.  

The model combines a **Susceptible–Infected–Recovered (SIR)** framework for the human population with a **Susceptible–Infected (SI)** framework for the mosquito population, implemented using **ordinary differential equations (ODEs)** solved in **R**.

---

## 📘 Model Overview

### Mathematical Framework
- **Humans:** ( Sₕ → Iₕ → Rₕ )  
- **Mosquitoes:** ( Sᵥ → Iᵥ )  
- The model dynamically adjusts mosquito–human transmission rates based on **ITN efficacy decay**, reflecting the buildup of **insecticide resistance** over time.

---

### 🧮 Key Equations

**1. ITN Efficacy Decay**

![E_t](https://latex.codecogs.com/svg.image?E_t%20=%20E_{initial}%20e^{-ResistanceRate%20(t/365)})

**2. Effective ITN Coverage**

![C_eff](https://latex.codecogs.com/svg.image?C_{eff}%20=%20ITN_{Coverage}%20\cdot%20E_t)

**3. Transmission Reduction Factor**

![T_red](https://latex.codecogs.com/svg.image?T_{red}%20=%201%20-%20C_{eff})

**4. Modified Transmission Rates**

![beta_vh](https://latex.codecogs.com/svg.image?\beta_{v%20\rightarrow%20h}%20=%20\beta_{v,\text{base}}%20\cdot%20T_{red}),  
![beta_hv](https://latex.codecogs.com/svg.image?\beta_{h%20\rightarrow%20v}%20=%20\beta_{h,\text{base}}%20\cdot%20T_{red})

---

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
