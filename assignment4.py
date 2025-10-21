"""
ATMS 523 Weather and Climate Data Analytics
EOF Analysis of Pacific Basin SST and Water Vapor

Author: Dara Procell
Date: October 20, 2025
Description: EOF analysis of Sea Surface Temperature anomalies and 
             correlation with total column water vapor over the Pacific Basin
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Detrending and Anomaly Functions 

def _time_as_float(time: xr.DataArray, time_dim: str) -> xr.DataArray:
    """Convert time to float (seconds since first timestamp)"""
    return (time - time.isel({time_dim: 0})).astype("timedelta64[s]").astype("int64").astype("float64")

def linear_detrend(obj: xr.DataArray | xr.Dataset, time_dim: str = "time") -> xr.DataArray | xr.Dataset:
    """
    Remove a linear trend y ~ s*(t - t̄_valid) + ȳ_valid at each grid point.
    Closed-form LS using reductions; dask-friendly; handles NaNs.
    """
    t = _time_as_float(obj[time_dim], time_dim)
    
    def _detrend_da(da: xr.DataArray) -> xr.DataArray:
        da = da.sortby(time_dim).astype("float32")
        if hasattr(da.data, "chunks"):
            da = da.chunk({time_dim: -1})
        mask = da.notnull()
        t_b = t.broadcast_like(da)
        t_mean_valid = t_b.where(mask).mean(time_dim, skipna=True)
        tc = t_b - t_mean_valid
        num = (da * tc).sum(time_dim, skipna=True)
        den = (tc**2).sum(time_dim, skipna=True)
        slope = xr.where(den > 0, num / den, 0.0)
        ybar = da.mean(time_dim, skipna=True)
        trend = slope * (t_b - t_mean_valid) + ybar
        return (da - trend).astype("float32")
    
    return obj.map(_detrend_da) if isinstance(obj, xr.Dataset) else _detrend_da(obj)

def monthly_anom_and_z(
    detr: xr.DataArray | xr.Dataset,
    time_dim: str = "time",
    base_period: tuple[str, str] | None = None,
    ddof: int = 1,
    eps: float = 1e-6,
):
    """
    From linearly-detrended data, remove monthly climatology and compute monthly z-scores.
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

# Main Analysis 

class PacificSSTEOFAnalysis:
    """
    Analyze Pacific Basin SST patterns using EOF analysis.
    """
    
    def __init__(self):
        """Initialize the analysis"""
        self.pacific_bounds = {
            'latitude': slice(65, -65),  # 65°N to 65°S
            'longitude': slice(120, 300)  # 120°E to 60°W (300°E)
        }
    
    def load_and_prepare_data(self, sst_file, tcwv_file, lsm_file=None):
        """
        Load ERA5 monthly mean data and prepare for analysis.
        
        Parameters:
        -----------
        sst_file : str
            Path to SST NetCDF file
        tcwv_file : str
            Path to total column water vapor NetCDF file
        lsm_file : str, optional
            Path to land-sea mask file
        """        
        ds_sst = xr.open_dataset(sst_file)
        ds_tcwv = xr.open_dataset(tcwv_file)
        
        # Identify time dimension (could be 'time' or 'valid_time')
        time_dim = 'valid_time' if 'valid_time' in ds_sst.dims else 'time'
        self.time_dim = time_dim
        
        # Select Pacific Basin region
        ds_sst = ds_sst.sel(**self.pacific_bounds)
        ds_tcwv = ds_tcwv.sel(**self.pacific_bounds)
        
        # Select time period (Jan 1979 - Dec 2024)
        ds_sst = ds_sst.sel({time_dim: slice('1979-01', '2024-12')})
        ds_tcwv = ds_tcwv.sel({time_dim: slice('1979-01', '2024-12')})
        
        # Apply land-sea mask 
        lsm = xr.open_dataset(lsm_file)
        lsm = lsm.sel(**self.pacific_bounds)
        ds_sst = ds_sst.where(lsm['lsm'] < 0.5)
        ds_tcwv = ds_tcwv.where(lsm['lsm'] < 0.5)
        
        # Combine into single dataset, rename variables to standard names
        sst_var = 'sst' if 'sst' in ds_sst.data_vars else list(ds_sst.data_vars)[0]
        tcwv_var = 'tcwv' if 'tcwv' in ds_tcwv.data_vars else list(ds_tcwv.data_vars)[0]
        
        self.ds = xr.Dataset({
            'sst': ds_sst[sst_var],
            'tcwv': ds_tcwv[tcwv_var]
        })
        print("Data loaded: " + str(len(self.ds[time_dim])) + " time steps")
        print("SST shape: " + str(self.ds['sst'].shape))
        print("TCWV shape: " + str(self.ds['tcwv'].shape))
        
        # Save original data for later
        self.ds_original = self.ds.copy()
        
    def process_anomalies(self):
        """
        Step 2: Detrend and deseasonalize the data, then standardize SST.
        """        
        # Chunk time dimension for efficiency
        self.ds = self.ds.chunk({self.time_dim: -1})
        
        # Step 1: Linear detrend
        detr = linear_detrend(self.ds[['sst', 'tcwv']], time_dim=self.time_dim)
        
        # Step 2: Remove monthly climatology and compute z-scores
        anom, z = monthly_anom_and_z(
            detr, 
            time_dim=self.time_dim,
            base_period=("1981-01-01", "2010-12-31")
        )
        
        # Store the detrended, deseasonalized data
        self.ds_anom = anom
        self.ds_z = z
        
        # Step 3: Standardize SST anomalies
        sst_anom = anom['sst']
        
        # Flatten spatial dimensions for standardization
        sst_mean = sst_anom.mean(dim=self.time_dim)
        sst_std = sst_anom.std(dim=self.time_dim)
        
        # Standardize: (X - mean) / std
        sst_standardized = (sst_anom - sst_mean) / sst_std
        
        self.sst_standardized = sst_standardized

        print("SST anomaly mean: " + format(float(sst_anom.mean()), '.6f'))
        print("SST anomaly std: " + format(float(sst_anom.std()), '.6f'))
    
    def perform_eof_analysis(self, n_eofs=10):
        """
        Step 3: Perform EOF analysis on standardized SST anomalies.
        
        Parameters:
        -----------
        n_eofs : int
            Number of EOFs to compute (default: 10)
        """        
        # Reshape data
        sst_data = self.sst_standardized
        lat_coords = sst_data.latitude.values
        lon_coords = sst_data.longitude.values
        
        # Flatten spatial dimensions and remove NaN grid points
        sst_2d = sst_data.stack(space=('latitude', 'longitude'))
        
        # Find valid (non-NaN) spatial points
        valid_mask = ~np.isnan(sst_2d.isel({self.time_dim: 0}))
        sst_valid = sst_2d[:, valid_mask.values]
        
        # Convert to numpy array and transpose to (n_samples, n_features)
        X = sst_valid.values.T  # Shape: (n_times, n_space)
        
        print("Data matrix shape: " + str(X.shape))
        print("Valid grid points: " + str(X.shape[1]))
        
        # PCA/ EOF analysis
        pca = PCA(n_components=n_eofs)
        pca.fit(X)
        
        # Get EOF patterns (principal components)
        eofs = pca.components_  # Shape: (n_eofs, n_space)
        
        # Get PC time series
        pcs = pca.transform(X)  # Shape: (n_times, n_eofs)
        
        # Get explained variance
        explained_var = pca.explained_variance_ratio_ * 100
        
        # Reconstruct full spatial fields for EOFs
        self.eofs_spatial = []
        for i in range(n_eofs):
            eof_full = np.full(valid_mask.shape, np.nan)
            eof_full[valid_mask.values] = eofs[i, :]
            eof_2d = eof_full.reshape(len(lat_coords), len(lon_coords))
            self.eofs_spatial.append(eof_2d)
        
        self.pca = pca
        self.pcs = pcs
        self.eofs = eofs
        self.explained_var = explained_var
        self.valid_mask = valid_mask
        self.lat_coords = lat_coords
        self.lon_coords = lon_coords

        print("Variance explained by first 5 EOFs: " + format(explained_var[:5].sum(), '.2f') + "%")

    def plot_eof_maps(self, n_eofs=5):
        """
        Step 3: Plot maps of the first n EOFs.
        
        Parameters:
        -----------
        n_eofs : int
            Number of EOFs to plot (default: 5)
        """        
        fig = plt.figure(figsize=(20, 12))
        
        for i in range(n_eofs):
            ax = fig.add_subplot(3, 2, i+1, projection=ccrs.PlateCarree(central_longitude=180))
            ax.set_extent([120, 300, -65, 65], ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.LAND, alpha=0.3)
            
            # Plot EOF
            eof_data = self.eofs_spatial[i]
            
            # Variable color scaling
            vmax = np.nanpercentile(np.abs(eof_data), 95)
            
            im = ax.contourf(
                self.lon_coords, self.lat_coords, eof_data,
                levels=np.linspace(-vmax, vmax, 21),
                cmap='RdBu_r', extend='both',
                transform=ccrs.PlateCarree()
            )
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, label='EOF Loading', shrink=0.8)
            ax.set_title(f'EOF {i+1} ({self.explained_var[i]:.2f}% variance)', fontsize=12, fontweight='bold')
            gl = ax.gridlines(draw_labels=True, alpha=0.3)
            gl.top_labels = False
            gl.right_labels = False
        
        plt.tight_layout()
        plt.savefig('pacific_sst_eof_maps.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_variance_explained(self, n_eofs=10):
        """
        Step 4: Plot percent variance explained by first n EOFs.
        
        Parameters:
        -----------
        n_eofs : int
            Number of EOFs to plot (default: 10)
        """        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Individual variance
        eof_numbers = np.arange(1, n_eofs + 1)
        ax1.bar(eof_numbers, self.explained_var[:n_eofs], color='steelblue', alpha=0.7)
        ax1.set_xlabel('EOF Number', fontsize=12)
        ax1.set_ylabel('Variance Explained (%)', fontsize=12)
        ax1.set_title('Variance Explained by Each EOF', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(eof_numbers)

        # Cumulative variance
        cumulative_var = np.cumsum(self.explained_var[:n_eofs])
        ax2.plot(eof_numbers, cumulative_var, 'o-', color='darkred', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of EOFs', fontsize=12)
        ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
        ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(eof_numbers)
        ax2.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90%')
        ax2.legend()
        plt.tight_layout()
        plt.savefig('eof_variance_explained.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def reconstruct_sst(self, n_eofs=5):
        """
        Step 5: Reconstruct SST field using first n EOFs.
        
        Parameters:
        -----------
        n_eofs : int
            Number of EOFs to use for reconstruction (default: 5)
        """        
        # Reconstruct using first n EOFs
        X_reconstructed = self.pcs[:, :n_eofs] @ self.eofs[:n_eofs, :]
        
        # Reshape back to spatial format
        reconstructed_full = np.full(
            (len(self.sst_standardized[self.time_dim]), *self.valid_mask.shape),
            np.nan
        )
        reconstructed_full[:, self.valid_mask.values] = X_reconstructed
        reconstructed_reshaped = reconstructed_full.reshape(
            len(self.sst_standardized[self.time_dim]),
            len(self.lat_coords),
            len(self.lon_coords)
        )
        
        # Create DataArray
        self.sst_reconstructed_std = xr.DataArray(
            reconstructed_reshaped,
            dims=[self.time_dim, 'latitude', 'longitude'],
            coords={
                self.time_dim: self.sst_standardized[self.time_dim],
                'latitude': self.lat_coords,
                'longitude': self.lon_coords
            }
        )
        
        # Unstandardize: multiply by std and add mean
        sst_anom = self.ds_anom['sst']
        sst_mean = sst_anom.mean(dim=self.time_dim)
        sst_std = sst_anom.std(dim=self.time_dim)
        
        sst_reconstructed_anom = self.sst_reconstructed_std * sst_std + sst_mean
        
        # Add back the trend and climatology to get observed values
        # For correlation we use anomalies
        self.sst_reconstructed = sst_reconstructed_anom
            
    def plot_reconstruction_correlation(self):
        """
        Step 5: Plot correlation between reconstructed and observed SST.
        """        
        # Calculate correlation at each grid point
        observed = self.ds_anom['sst']
        reconstructed = self.sst_reconstructed
        
        # Correlation computation
        correlation = xr.corr(observed, reconstructed, dim=self.time_dim)
        
        # Plot
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
        ax.set_extent([120, 300, -65, 65], ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.LAND, alpha=0.3)
        im = ax.contourf(
            self.lon_coords, self.lat_coords, correlation.values,
            levels=np.linspace(0, 1, 21),
            cmap='YlOrRd', extend='min',
            transform=ccrs.PlateCarree()
        )
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, label='Pearson Correlation Coefficient', shrink=0.8)
        ax.set_title('Correlation: Reconstructed (5 EOFs) vs Observed SST Anomalies', fontsize=14, fontweight='bold')
        gl = ax.gridlines(draw_labels=True, alpha=0.3)
        gl.top_labels = False
        gl.right_labels = False
        
        plt.tight_layout()
        plt.savefig('sst_reconstruction_correlation.png', dpi=300, bbox_inches='tight')
        print("Mean correlation: " + format(float(correlation.mean()), '.3f'))
        plt.show()
        
        self.reconstruction_corr = correlation
    
    def analyze_sst_tcwv_correlation(self):
        """
        Step 6: Compute correlation between SST EOF1 and total column water vapor.
        """        
        # Get PC1 time series
        pc1 = self.pcs[:, 0]
        
        # Create DataArray for PC1
        pc1_da = xr.DataArray(
            pc1,
            dims=[self.time_dim],
            coords={self.time_dim: self.sst_standardized[self.time_dim]}
        )
        
        # Get detrended, deseasonalized, standardized TCWV
        tcwv_anom = self.ds_z['tcwv']
        
        # Correlation computation
        correlation = xr.corr(pc1_da, tcwv_anom, dim=self.time_dim)
        
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
        ax.set_extent([120, 300, -65, 65], ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.LAND, alpha=0.2)
        
        # Plot correlation (don't mask over land)
        vmax = 0.8
        im = ax.contourf(
            self.lon_coords, self.lat_coords, correlation.values,
            levels=np.linspace(-vmax, vmax, 21),
            cmap='RdBu_r', extend='both',
            transform=ccrs.PlateCarree()
        )
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05,label='Pearson Correlation Coefficient', shrink=0.8)
        ax.set_title(f'Correlation: SST EOF1 ({self.explained_var[0]:.2f}% var) vs Total Column Water Vapor', fontsize=14, fontweight='bold')
        gl = ax.gridlines(draw_labels=True, alpha=0.3)
        gl.top_labels = False
        gl.right_labels = False
        
        plt.tight_layout()
        plt.savefig('sst_eof1_tcwv_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.sst_tcwv_corr = correlation
    
    def run_complete_analysis(self, sst_file, tcwv_file, lsm_file=None):
        """
        Run the complete analysis workflow.
        
        Parameters:
        -----------
        sst_file : str
            Path to SST NetCDF file
        tcwv_file : str
            Path to total column water vapor NetCDF file
        lsm_file : str, optional
            Path to land-sea mask file
        """
        
        # Step 1: Load and prepare data
        self.load_and_prepare_data(sst_file, tcwv_file, lsm_file)
        
        # Step 2: Process anomalies
        self.process_anomalies()
        
        # Step 3: Perform EOF analysis
        self.perform_eof_analysis(n_eofs=10)
        
        # Step 3: Plot EOF maps
        self.plot_eof_maps(n_eofs=5)
        
        # Step 4: Plot variance explained
        self.plot_variance_explained(n_eofs=10)
        
        # Step 5: Reconstruct SST and plot correlation
        self.reconstruct_sst(n_eofs=5)
        self.plot_reconstruction_correlation()
        
        # Step 6: Analyze SST-TCWV correlation
        self.analyze_sst_tcwv_correlation()


# Main function

if __name__ == "__main__":
    analyzer = PacificSSTEOFAnalysis()
    
    # YOU NEED TO DOWNLOAD THESE FROM COPERNICUS FIRST
    sst_file = "era5_sst_monthly_1979-2024.nc"
    tcwv_file = "era5_tcwv_monthly_1979-2024.nc"
    lsm_file = "era5_land_sea_mask.nc" 
    
    analyzer.run_complete_analysis(sst_file, tcwv_file, lsm_file)

"""
Data Download Instructions:
---------------------------
1. Go to: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means
2. Download the following variables for Jan 1979 - Dec 2024:
   - Sea surface temperature (sst)
   - Total column water vapour (tcwv)
   - Land-sea mask (lsm) - optional but recommended
3. Geographic area: 65°N to 65°S, 120°E to 300°E (or -60°W)
4. Time: Monthly means, all months, 1979-2024
5. Format: NetCDF

References:
-----------
Hersbach, H., et al. (2020). The ERA5 global reanalysis. 
Quarterly Journal of the Royal Meteorological Society, 146(730), 1999-2049.
"""