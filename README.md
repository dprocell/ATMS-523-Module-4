# Assignment 4 code walkthrough


1. Load Data

- Load ERA5 NetCDF file with SST, TCWV, and land-sea mask

2. Detrend and Compute Anomalies

- Remove linear trends from SST and TCWV at each grid point
- Remove seasonal cycle (monthly climatology) to get anomalies
- Standardize anomalies (z-scores: subtract mean, divide by std)

3. EOF Analysis

- Mask ocean-only points using land-sea mask
- Stack 2D spatial grid into 1D array
- Perform SVD to extract EOFs (spatial patterns) and PCs (time series)
- Calculate variance explained by each EOF
- Plot first 5 EOF spatial patterns

4. Variance Explained

- Plot bar chart showing variance explained by first 10 EOFs
- Add cumulative variance line

5. Reconstruct SST

- Rebuild SST using only first 5 EOFs
- Reverse standardization (multiply by std, add mean)
- Add linear trends back
- Compute correlation between reconstructed and observed SST
- Plot correlation map

6. SST-TCWV Teleconnection

- Correlate EOF1 principal component with TCWV anomalies at each grid point
- Plot correlation map to reveal atmospheric moisture patterns linked to dominant SST mode