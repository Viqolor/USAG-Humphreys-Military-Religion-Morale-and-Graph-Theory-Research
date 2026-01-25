# USAG-Humphreys-Military-Religion-Morale-and-Graph-Theory-Research

*Disclaimer: The views and opinions expressed under this research are strictly of our own and do not necessarily represent the official position or policy of the 2nd Infantry Division, the Eighth Army, USFK, and ROK Ministry of National Defense.*


**Project Overview**


This repository contains a Python-based simulation engine designed to quantify the impact of religious facilities on garrison morale. Using a bipartite graph model and Monte Carlo simulations (10,000 iterations), this project forecasts how shifts in infrastructure affect 10 specific barrack nodes.The research evaluates morale through four experimental lenses:


Experiment 1 (Baseline): Current state of the garrison.  

Experiment 2 (Removal): Impact of removing all religious influence.  

Experiment 3 (Swaps): Iterative testing of switching religious facilities to be secular.  

Experiment 4 (Additions): Strategic placement of a new high-intensity religious facilit.

===========================================================================

**Mathematical Framework**

Edge weights in our bipartite graph model is calculated using the following multi-component function:  

$$W_{ur} = e^{-\alpha D} \cdot (1 + \beta \cdot Q) \cdot \frac {I}{B}$$ 

$D$: Distance between facility and barracks.  

$Q$: Quality of facility (derived from service intensity).  

$\alpha/\beta$: Constants for spatial decay and quality weighting.  

$I$: Facility intensity (weekly usage in person-hours).

$B$: Standard Barrack Unit

===========================================================================

**Installation & Usage**


*Prerequisites*

Python 3.8+  

Dependencies: pandas, numpy, scipy, seaborn (optional for plotting)


*Running the Simulation*  

Clone the repository: clone https://github.com/Viqolor/USAG-Humphreys-Military-Religion-Morale-and-Graph-Theory-Research.git  

Execute the control panel to run all experiments and significance tests: python control_panel.py

===========================================================================

**File Structure**

research_engine.py: Core logic for the Monte Carlo simulation and significance testing.  

control_panel.py: Script to execute the four experiments and export results.  

master_results.csv: Summary of means, P-values, Cohen's $d$, and 95% Confidence Intervals for all experimental scenarios.  

distances.csv / facilities.csv / blocks.csv: Input datasets defining spatial relationships and facility/unit attributes.

===========================================================================

**Significance Testing**

Each experiment is compared against the baseline using:  

Two-Sample T-Tests: To determine P-values for both global and node-level shifts.  

Cohen's $d$: To quantify effect size (Negligible, Small, Medium, Large).  

95% Confidence Intervals: For the mean difference across 10,000 simulations.

===========================================================================

**Data Visualization**

The raw simulation data is exported into individual CSV files (e.g., raw_dist_Baseline.csv) designed for box plot generation. These files are sampled at $n=2000$ per node to ensure high-performance rendering in tools like RAWGraphs or Seaborn.

===========================================================================

**How to Cite**

If you use this code in your research, please cite: Kim, Seung-rae, "Modeling Spiritual Infrastructure: A Spatial Analysis of Religious Facility Impact on USAG Humphreys Morale through Monte Carlo Simulation," (2026).
