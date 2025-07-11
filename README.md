# HYSETS-Data-Process
Code for creating input data to ML/DL models for HYSETS catchments

# HYSETS Input Processing for ML/DL Applications

This repository provides streamlined tools for preparing machine learning and deep learning input datasets using the **HYSETS** database, **Daymet** climate data, and **in-situ gauge station** measurements.

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

---
## Reference

Thornton, M.M., R. Shrestha, Y. Wei, P.E. Thornton, S-C. Kao, and B.E. Wilson. 2022. Daymet: Daily Surface Weather Data on a 1-km Grid for North America, Version 4 R1. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/2129

Arsenault, R., Brissette, F., Martel, J.-L., Troin, M., Lévesque, G., Davidson-Chaput, J., … Poulin, A. (2024, October 1).
HYSETS - A 14425 watershed Hydrometeorological Sandbox over North America. https://doi.org/10.17605/OSF.IO/RPC3W
             
Kratzert, F. (2022). Caravan ‐ A global community dataset for large‐sample hydrology. Zenodo. https://doi.org/10.5281/zenodo.7540792
    


