# Monte Carlo Protein Folding — HP Model Simulation

## Introduction

This project explores protein structure prediction through **Monte Carlo simulations using the Hydrophobic–Polar (HP) lattice model**.  
The HP model simplifies proteins into sequences of hydrophobic (H) and polar (P) residues placed on a 2D or 3D lattice. Folding is simulated through random conformational moves evaluated via an energy function based on H–H interactions and accepted using the Metropolis criterion.

The goals of this project are to:

- simulate protein folding on **2D and 3D lattices**,  
- track **energy minimization** during folding,  
- visualize folding trajectories as animations,  
- generate and compare final conformations,  
- provide a fully reproducible environment using **Pixi**.

This repository includes two scripts (`hp_2d.py` and `hp_3d.py`) that run the simulations, generate videos, and save outputs in structured folders.

## Prerequisites

Before starting, make sure you have **Pixi** installed:  
[Pixi Installation Guide](https://pixi.sh/dev/installation/)

You must also have **git** installed on your machine.

---

## Installation and Synchronization

Clone this repository and move into the project folder:

```bash
git clone https://github.com/anaisdlss/Monte-Carlo-Pixi-Projet.git Delassus_MonteCarlo
cd Delassus_MonteCarlo
```
Then synchronize the Pixi environment for the project:
```bash
pixi install
```
---

## Usage

### 1. Running the 2D Simulation

Start by running the **hp_2d.py** script:

```bash
pixi run python hp_2d.py
```
Follow the on-screen instructions.
Execution should take a few dozen seconds.
Make sure to close the generated video window so that the script can finish properly.
All generated outputs will appear in the **out2d/** directory.

### 2. Running the 3D Simulation

Next, launch the **hp_3d.py** script:

```bash
pixi run python hp_3d.py
```
Follow the instructions.
This execution may take a few minutes.
Again, close the generated video window to allow the script to complete.

Outputs will be stored in the **out3d/** directory.


