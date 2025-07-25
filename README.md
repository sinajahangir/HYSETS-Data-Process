# HYSETS-Data-Process
Code for creating input data to ML/DL models for HYSETS catchments, and LSTM model development for streamflow and water level prediction.
The codes provided are general purpose (can be tweaked to be used for all HYSETS catchments), but have been explicitly developed for gauge 02GA014 in Ontario, CA

# HYSETS Input Processing for ML/DL Applications

This repository provides streamlined tools for preparing machine learning and deep learning input datasets using the **HYSETS** database (Arsenault et al., 2024), **Daymet** climate data (Thornton et al., 2022), and **in-situ gauge station** measurements.

---

## Overview

The pipeline is tailored for data-driven hydrological modeling and predictive analytics. It integrates:

- **Dynamic Forcings**
  - Daily precipitation, minimum and maximum temperature, and solar radiation from **Daymet**
  - In-situ **water level** and **streamflow** data from gauge stations

- **Static Features**
  - Some watershed characteristics from **HYSETS** (e.g., land use, elevation, slope)
  - **Soil types, land cover**, and other geophysical features via **Caravan** and HYSETS

---

## Features

- ✅ Automated retrieval and formatting of Daymet data (via GEE or cloud)
- ✅ Integration of gauge station observations for supervised ML tasks
- ✅ Extraction and preprocessing of static geospatial descriptors
- ✅ Ready-to-use input structure for sequence-based DL models
- ✅ DL fine-tuning for efficient and effective hydrological prediction

---

## Fine-Tuning

Sample code is provided for fine-tuning an LSTM model trained on >390 catchments in CONUS for a single catchment in Canada

## Meta-Learning

A probabilistic meta-learner (e.g., Gaussian) is developed for combining the water level forecast of the foundation model and the LSTM simulation model

---
## Reference

Thornton, M.M., R. Shrestha, Y. Wei, P.E. Thornton, S-C. Kao, and B.E. Wilson. 2022. Daymet: Daily Surface Weather Data on a 1-km Grid for North America, Version 4 R1. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/2129

Arsenault, R., Brissette, F., Martel, J.-L., Troin, M., Lévesque, G., Davidson-Chaput, J., … Poulin, A. (2024, October 1).
HYSETS - A 14425 watershed Hydrometeorological Sandbox over North America. https://doi.org/10.17605/OSF.IO/RPC3W
             
Kratzert, F. (2022). Caravan ‐ A global community dataset for large‐sample hydrology. Zenodo. https://doi.org/10.5281/zenodo.7540792
    


