# NN-KPI-Predictor

Neural-network surrogate model for predicting business process KPIs from simulation scenarios.

## Overview

This project explores how **machine learning can accelerate business process optimization** by learning the relationship between process configuration parameters and resulting performance indicators.

Traditional BPMN what-if analysis requires running many computationally expensive simulations. This repository implements a **surrogate modeling approach** where neural networks are trained on simulation-generated data to predict key process KPIs such as:

- **Cycle time**
- **Waiting time**
- **Operational cost**

The project was developed as part of my Master's thesis in Computer Engineering at **Sapienza University of Rome**.

---

## Motivation

Business process analysts often need to evaluate thousands of alternative scenarios:

- number of workers
- resource allocation
- salaries
- machine availability
- task durations
- branching probabilities

Running a full simulation for every candidate configuration is slow. The goal of this project is to **replace repeated simulations with a fast neural-network predictor**, enabling near-instant KPI estimation and more efficient process optimization.

---

## Pipeline

<pre>
Event logs / BPMN process
          │
          ▼
Simulation engine (SimPy / process simulation)
          │
          ▼
Scenario generation
          │
          ▼
KPI dataset creation
          │
          ▼
Feature preprocessing
          │
          ▼
Neural-network training
          │
          ▼
KPI prediction
</pre>

---

## Repository Structure

<pre>
NN-KPI-Predictor/
├── data/                  # Generated simulation datasets
├── models/                # Neural-network architectures
├── training/              # Training scripts
├── evaluation/            # Metrics and analysis
├── utils/                 # Helper functions
├── requirements.txt
└── README.md
</pre>

---

## Features

- Multi-output KPI prediction
- Configurable neural-network architectures
- Residual blocks and normalization layers
- Dataset merging and preprocessing utilities
- Evaluation with MAE, MSE, and MedAE
- Support for large synthetic process-simulation datasets

---

## Model

The main architecture is a **deep feed-forward surrogate network** with:

- Linear embedding layer
- Layer normalization
- Mish activation
- Residual blocks
- Shared latent representation
- Separate prediction heads for each KPI

This design allows the model to learn common process dynamics while specializing for each target metric.

---

## Example Use Case

Given a process scenario:

<Code value=
