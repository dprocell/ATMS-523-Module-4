"""
ATMS 523 Weather and Climate Data Analytics
Project 4

Author: Dara Procell
Date: October 21, 2025
Description: Era5 EOF analysis for SST and TCWV with land-sea masking
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import Union, Tuple, Optional

# STEP 1: Load the data

ds = xr.open_dataset('era5_data.nc')
ds = ds.rename({'tcrw': 'tcwv'})
time_dim = 'valid_time'

print(ds)

# DETRENDING AND ANOMALY FUNCTIONS 

def _time_as_float(time: xr.DataArray, time_dim: str) -> xr.DataArray:
    # numeric seconds since first timestamp (keeps numbers small)
    return (time - time.isel({time_dim: 0})).astype("timedelta64[s]").astype("int64").astype("float64")

def linear_detrend(obj: Union[xr.DataArray, xr.Dataset], time_dim: str = "time") -> Union[xr.DataArray, xr.Dataset]:
    """
    Remove a linear trend y ~ s*(t - t̄_valid) + ȳ_valid at each grid point.
    Closed-form LS using reductions; dask-friendly; handles NaNs.
    """
    t = _time_as_float(obj[time_dim], time_dim)  # (time,)
    def _detrend_da(da: xr.DataArray) -> xr.DataArray:
        da = da.sortby(time_dim).astype("float32")
        # Skip chunking if dask is not available
        # if hasattr(da.data, "chunks"):
        #     da = da.chunk({time_dim: -1})  # one chunk along time
        mask = da.notnull()                                # (time, ...)
        t_b = t.broadcast_like(da)                         # (time, ...)
        t_mean_valid = t_b.where(mask).mean(time_dim, skipna=True)
        tc = t_b - t_mean_valid                            # centered time per point
        num = (da * tc).sum(time_dim, skipna=True)
        den = (tc**2).sum(time_dim, skipna=True)
        slope = xr.where(den > 0, num / den, 0.0)
        ybar  = da.mean(time_dim, skipna=True)
        trend = slope * (t_b - t_mean_valid) + ybar
        return (da - trend).astype("float32")
    return obj.map(_detrend_da) if isinstance(obj, xr.Dataset) else _detrend_da(obj)

def monthly_anom_and_z(
    detr: Union[xr.DataArray, xr.Dataset],
    time_dim: str = "time",
    base_period: Optional[Tuple[str, str]] = None,
    ddof: int = 1,
    eps: float = 1e-6,
):
    """
    From linearly-detrended data, remove monthly climatology and compute monthly z-scores.f_spatial
    Returns (anom, z). Works for Dataset or DataArray.
    """
    clim_src = detr if base_period is None else detr.sel({time_dim: slice(*base_period)})
    key = f"{time_dim}.month"

    clim_mean = clim_src.groupby(key).mean(time_dim, skipna=True)
    anom = detr.groupby(key) - clim_mean

    clim_std = clim_src.groupby(key).std(time_dim, skipna=True, ddof=ddof)
    safe_std = xr.where(clim_std > eps, clim_std, np.nan)
    z = anom.groupby(key) / safe_std
    return anom, z


# STEP 2: Compute anomalies (deseasonalize and detrend)

# Detrend SST and TCWV
detr = linear_detrend(ds[["sst", "tcwv"]], time_dim=time_dim)

# Compute monthly anomalies and standardized anomalies (z-scores)
# Use 1981-2010 as base period (standard WMO climatology) or None for full period
anom, z = monthly_anom_and_z(detr, time_dim=time_dim,
                             base_period=("1981-01-01", "2010-12-31"))

print("Anomalies:", anom)
print("Z-scores:", z)

# SST standardized anomalies are already in z['sst']
sst_std = z['sst']  # This is already standardized (mean=0, std=1)


# STEP 3: EOF Analysis on SST anomalies

# Mask ocean-only points using land-sea mask
lsm_static = ds['lsm'].isel({time_dim: 0})
ocean_mask = lsm_static < 0.5  # ocean points

# Apply ocean mask to SST standardized anomalies
sst_ocean = sst_std.where(ocean_mask)

# Stack spatial dimensions for EOF analysis
sst_stacked = sst_ocean.stack(space=('latitude', 'longitude'))
sst_stacked = sst_stacked.dropna(dim='space', how='all')

# Convert to numpy array for EOF computation
sst_array = sst_stacked.values.T 

# Remove any remaining NaNs (fill with 0 for EOF, or drop those points)
valid_mask = ~np.isnan(sst_array).any(axis=1)
sst_clean = sst_array[valid_mask, :]
space_coords_valid = sst_stacked.space.values[valid_mask]


# SVD for EOF analysis
U, S, Vt = np.linalg.svd(sst_clean.T, full_matrices=False)

# EOFs are the spatial patterns (rows of Vt), PCs are the time series
eofs_space = Vt  # shape: (n_eofs, space)
pcs_time = U * S  # shape: (time, n_eofs)

total_var = np.sum(S**2)
var_explained = (S**2) / total_var * 100

# Plot first 5 EOFs
fig, axes = plt.subplots(3, 2, figsize=(14, 12), 
                         subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes.flatten()

for i in range(5):
    ax = axes[i]
    
    # Reconstruct 2D field for this EOF
    eof_field = np.full(len(sst_stacked.space), np.nan)
    eof_field[valid_mask] = eofs_space[i, :]
    
    # Unstack to original grid
    eof_2d = xr.DataArray(
        eof_field,
        coords={'space': sst_stacked.space},
        dims=['space']
    ).unstack('space')
    
    # Plot
    vmin = np.nanpercentile(eof_2d.values, 2)
    vmax = np.nanpercentile(eof_2d.values, 98)
    vlim = max(abs(vmin), abs(vmax))
    levels = np.linspace(-vlim, vlim, 21)
    im = ax.contourf(eof_2d.longitude, eof_2d.latitude, eof_2d,
                     levels=levels, cmap='RdBu_r', transform=ccrs.PlateCarree(),
                     extend='both')
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_title(f'EOF {i+1} ({var_explained[i]:.2f}%)', fontsize=12)
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)

axes[5].remove()

plt.tight_layout()
plt.savefig('eof_spatial_patterns.png', dpi=300, bbox_inches='tight')
plt.show()


# STEP 4: Plot variance explained by first 10 EOFs

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(1, 11), var_explained[:10], color='steelblue', edgecolor='black')
ax.set_xlabel('EOF Number', fontsize=12)
ax.set_ylabel('Variance Explained (%)', fontsize=12)
ax.set_title('Variance Explained by First 10 EOFs', fontsize=14, fontweight='bold')
ax.set_xticks(range(1, 11))
ax.grid(axis='y', alpha=0.3)
ax2 = ax.twinx()
cumsum = np.cumsum(var_explained[:10])
ax2.plot(range(1, 11), cumsum, 'ro-', linewidth=2, markersize=8, label='Cumulative')
ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.legend(loc='lower right')
plt.tight_layout()
plt.savefig('variance_explained.png', dpi=300, bbox_inches='tight')
plt.show()


# STEP 5: Reconstruct SST using first 5 EOFs and compute correlation

# Reconstruct using first 5 EOFs
n_eofs = 5
sst_reconstructed_std = pcs_time[:, :n_eofs] @ eofs_space[:n_eofs, :]

# Convert back to 2D
sst_recon_field = np.full((len(sst_stacked.space), len(sst_stacked[time_dim])), np.nan)
sst_recon_field[valid_mask, :] = sst_reconstructed_std.T

# Unstack to grid
sst_recon_2d = xr.DataArray(
    sst_recon_field,
    coords={
        'space': sst_stacked.space,
        time_dim: sst_stacked[time_dim]
    },
    dims=['space', time_dim]
).unstack('space')

# "Unstandardize" - reverse the z-score transformation
# z = (x - mean) / std, so x = z * std + mean
# Get monthly means and stds from the original detrended data
detr_sst = detr['sst']
key = f"{time_dim}.month"
monthly_mean = detr_sst.sel({time_dim: slice("1981-01-01", "2010-12-31")}).groupby(key).mean(time_dim, skipna=True)
monthly_std = detr_sst.sel({time_dim: slice("1981-01-01", "2010-12-31")}).groupby(key).std(time_dim, skipna=True)

# Apply unstandardization: x = z * std + mean
# We need to do this with groupby
sst_recon_unstd = xr.zeros_like(sst_recon_2d)
for month in range(1, 13):
    month_mask = sst_recon_2d[time_dim].dt.month == month
    sst_recon_unstd[{time_dim: month_mask}] = (
        sst_recon_2d[{time_dim: month_mask}] * monthly_std.sel(month=month) + monthly_mean.sel(month=month)
    )

# Add back the trend
# Need to recompute the original trend that was removed
t = _time_as_float(ds[time_dim], time_dim)
da_orig = ds['sst'].astype("float32")
mask = da_orig.notnull()
t_b = t.broadcast_like(da_orig)
t_mean_valid = t_b.where(mask).mean(time_dim, skipna=True)
tc = t_b - t_mean_valid
num = (da_orig * tc).sum(time_dim, skipna=True)
den = (tc**2).sum(time_dim, skipna=True)
slope = xr.where(den > 0, num / den, 0.0)
ybar = da_orig.mean(time_dim, skipna=True)
trend = slope * (t_b - t_mean_valid) + ybar
sst_recon_final = sst_recon_unstd + trend

correlation = xr.corr(sst_recon_final, ds['sst'], dim=time_dim)

# Plot
fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={'projection': ccrs.PlateCarree()})
corr_min = np.nanmin(correlation.values)
corr_max = np.nanmax(correlation.values)
print("Correlation range: " + "{}".format(corr_min) + " to " + "{}".format(corr_max))


# If else for correct mapping of colors
if corr_min > 0:
    levels = np.linspace(max(0.7, corr_min), 1.0, 21)
    im = ax.contourf(correlation.longitude, correlation.latitude, correlation,
                     levels=levels, cmap='Reds',
                     transform=ccrs.PlateCarree(), extend='min')
else:
    # If there are negative correlations, use symmetric diverging scale
    vlim = max(abs(corr_min), abs(corr_max))
    levels = np.linspace(-vlim, vlim, 21)
    im = ax.contourf(correlation.longitude, correlation.latitude, correlation,
                     levels=levels, cmap='RdBu_r',
                     transform=ccrs.PlateCarree(), extend='both')
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.set_title('Correlation: Reconstructed (5 EOFs) vs Observed SST', fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
cbar.set_label('Correlation Coefficient', fontsize=12)
plt.tight_layout()
plt.savefig('sst_reconstruction_correlation.png', dpi=300, bbox_inches='tight')
plt.show()


# STEP 6: Correlation between SST EOF1 and TCWV anomalies

pc1 = pcs_time[:, 0]
tcwv_std = z['tcwv']

# Compute correlation at each grid point
# PC1 is 1D (time), tcwv_std is 3D (time, lat, lon)
# Create a DataArray for PC1 
pc1_da = xr.DataArray(pc1, coords={time_dim: sst_stacked[time_dim]}, dims=[time_dim])
correlation_tcwv = xr.corr(pc1_da, tcwv_std, dim=time_dim)

# Plot
fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={'projection': ccrs.PlateCarree()})
im = ax.contourf(correlation_tcwv.longitude, correlation_tcwv.latitude, correlation_tcwv,
                 levels=np.linspace(-0.8, 0.8, 17), cmap='RdBu_r',
                 transform=ccrs.PlateCarree(), extend='both')
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.set_title('Correlation: SST EOF1 PC vs TCWV Anomalies', fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
cbar.set_label('Correlation Coefficient', fontsize=12)
plt.tight_layout()
plt.savefig('eof1_tcwv_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

