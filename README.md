# 🦟Simulating and Fitting Malaria Transmission Model in Madagascar: Impact of Insecticide-Treated Nets (ITNs)

This study investigates the epidemiological impact of ITNs under varying resistance scenarios. Using a modified Susceptible, Infected, Recovered (SIR) model with integrated vector dynamics, we simulate malaria transmission in Madagascar and compare outcomes across four scenarios:

- **No ITNs**
- **Standard ITNs (No Resistance)**
- **ITNs with Low Resistance**
- **ITNs with High Resistance**


By combining modeling and recent prevalence data (2019–2024), the analysis highlights the effectiveness of ITNs, the impact of resistance and implications for malaria control strategies in Madagascar.


This repository contains a **deterministic malaria transmission model** that simulates the interaction between humans and mosquito vectors under varying **Insecticide-Treated Net (ITN)** coverage and resistance conditions.  

The model combines a **Susceptible–Infected–Recovered (SIR)** framework for the human population with a **Susceptible–Infected (SI)** framework for the mosquito population, implemented using **ordinary differential equations (ODEs)** solved in **R**.

---

## 📘 Model Overview

### Mathematical Framework
- **Humans:** ( Sₕ → Iₕ → Rₕ )  
- **Mosquitoes:** ( Sᵥ → Iᵥ )  
- The model dynamically adjusts mosquito–human transmission rates based on **ITN efficacy decay**, reflecting the buildup of **insecticide resistance** over time.

---


For a complete explanation of the model equations, parameter definitions, and assumptions, see the accompanying article on **[Medium](https://medium.com/@freshsafoduker300/simulating-and-fitting-malaria-transmission-model-in-madagascar-impact-of-insecticide-treated-nets-fd9c10d4cda4)**.


![Made with R](https://img.shields.io/badge/Made%20with-R-276DC3?style=for-the-badge&logo=r&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)
[![Medium Article](https://img.shields.io/badge/Read%20on-Medium-black?style=for-the-badge&logo=medium)](https://medium.com/@freshsafoduker300/simulating-and-fitting-malaria-transmission-model-in-madagascar-impact-of-insecticide-treated-nets-fd9c10d4cda4)
[![GitHub repo](https://img.shields.io/badge/View%20on-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/Nana-Safo-Duker/Malaria_Transmission_Model_Madagascar)


---

## 🧩 Files

| File | Description |
|------|--------------|
| `Malaria_Transmission_Model_Madagascar.r` | Main R script implementing the SIR–SI model and simulations. |
| `Malaria-Transmission-Model-in-Madagascar.ipynb` | Jupyter notebook version (R kernel) for interactive exploration. |

---

## ⚙️ Requirements

- **R version ≥ 4.0**
- Required packages:
  ```r
  install.packages(c("deSolve", "tidyverse"))
