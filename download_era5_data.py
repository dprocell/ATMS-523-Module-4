## download data

"""
Download ERA5 Monthly Mean Data for Pacific Basin SST EOF Analysis
Using Copernicus Climate Data Store (CDS) API

Setup Instructions:
1. Create account at: https://cds.climate.copernicus.eu/
2. Get your API key from: https://cds.climate.copernicus.eu/how-to-api
3. Install the CDS API client: pip install cdsapi
"""

import cdsapi
import xarray as xr

# Initialize the CDS API client
# Make sure you have set up your ~/.cdsapirc file with your credentials
c = cdsapi.Client()

def download_era5_sst():
    """
    Download Sea Surface Temperature monthly means from ERA5
    Jan 1979 - Dec 2024, Pacific Basin (65°N to 65°S, 120°E to 60°W)
    """
    print("Downloading Sea Surface Temperature data...")
    
    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': 'sea_surface_temperature',
            'year': [
                '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986',
                '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994',
                '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002',
                '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010',
                '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018',
                '2019', '2020', '2021', '2022', '2023', '2024',
            ],
            'month': [
                '01', '02', '03', '04', '05', '06',
                '07', '08', '09', '10', '11', '12',
            ],
            'time': '00:00',
            'area': [
                65, 120, -65, 300,  # North, West, South, East
            ],
            'format': 'netcdf',
        },
        'era5_sst_monthly_1979-2024.nc'
    )
    print("SST download complete!")


def download_era5_tcwv():
    """
    Download Total Column Water Vapor monthly means from ERA5
    Jan 1979 - Dec 2024, Pacific Basin (65°N to 65°S, 120°E to 60°W)
    """
    print("Downloading Total Column Water Vapor data...")
    
    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': 'total_column_water_vapour',
            'year': [
                '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986',
                '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994',
                '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002',
                '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010',
                '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018',
                '2019', '2020', '2021', '2022', '2023', '2024',
            ],
            'month': [
                '01', '02', '03', '04', '05', '06',
                '07', '08', '09', '10', '11', '12',
            ],
            'time': '00:00',
            'area': [
                65, 120, -65, 300,  # North, West, South, East
            ],
            'format': 'netcdf',
        },
        'era5_tcwv_monthly_1979-2024.nc'
    )
    print("TCWV download complete!")


def download_era5_land_sea_mask():
    """
    Download Land-Sea Mask from ERA5
    This is invariant data (doesn't change with time)
    """
    print("Downloading Land-Sea Mask...")
    
    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': 'land_sea_mask',
            'year': '2020',  # Any year will do since it's invariant
            'month': '01',   # Any month will do
            'time': '00:00',
            'area': [
                65, 120, -65, 300,  # North, West, South, East
            ],
            'format': 'netcdf',
        },
        'era5_land_sea_mask.nc'
    )
    print("Land-Sea Mask download complete!")


def download_all_data():
    """
    Download all required data for the EOF analysis
    """
    print("="*70)
    print("DOWNLOADING ERA5 DATA FOR PACIFIC BASIN SST EOF ANALYSIS")
    print("="*70)
    print("\nRegion: 65°N to 65°S, 120°E to 300°E (60°W)")
    print("Period: January 1979 - December 2024")
    print("\nThis may take 10-30 minutes depending on the queue...")
    print("="*70 + "\n")
    
    # Download each dataset
    try:
        download_era5_sst()
        print("\n" + "-"*70 + "\n")
        
        download_era5_tcwv()
        print("\n" + "-"*70 + "\n")
        
        download_era5_land_sea_mask()
        print("\n" + "-"*70 + "\n")
        
        print("\n" + "="*70)
        print("ALL DOWNLOADS COMPLETE!")
        print("="*70)
        print("\nFiles created:")
        print("  - era5_sst_monthly_1979-2024.nc")
        print("  - era5_tcwv_monthly_1979-2024.nc")
        print("  - era5_land_sea_mask.nc")
        print("\nYou can now run the EOF analysis script!")
        
    except Exception as e:
        print(f"\nError during download: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you've accepted the terms at:")
        print("   https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means")
        print("2. Check your API credentials in ~/.cdsapirc")
        print("3. Verify your account is activated")


def verify_downloads():
    """
    Verify the downloaded files and print basic info
    """
    print("\n" + "="*70)
    print("VERIFYING DOWNLOADED FILES")
    print("="*70 + "\n")
    
    files = [
        'era5_sst_monthly_1979-2024.nc',
        'era5_tcwv_monthly_1979-2024.nc',
        'era5_land_sea_mask.nc'
    ]
    
    for filename in files:
        try:
            print(f"Checking {filename}...")
            ds = xr.open_dataset(filename)
            print(f"  Variables: {list(ds.data_vars)}")
            print(f"  Dimensions: {dict(ds.dims)}")
            print(f"  Coordinates: {list(ds.coords)}")
            
            # Print time range if applicable
            if 'time' in ds.dims:
                print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
            elif 'valid_time' in ds.dims:
                print(f"  Time range: {ds.valid_time.values[0]} to {ds.valid_time.values[-1]}")
            
            print(f"  ✓ File verified successfully!\n")
            ds.close()
            
        except FileNotFoundError:
            print(f"  ✗ File not found!\n")
        except Exception as e:
            print(f"  ✗ Error: {e}\n")


# ============================================================================
# SETUP INSTRUCTIONS
# ============================================================================

def setup_cds_api():
    """
    Print instructions for setting up the CDS API
    """
    print("""
    ═══════════════════════════════════════════════════════════════════════
    CDS API SETUP INSTRUCTIONS
    ═══════════════════════════════════════════════════════════════════════
    
    1. CREATE AN ACCOUNT
       Go to: https://cds.climate.copernicus.eu/
       Click "Register" and create a free account
    
    2. ACCEPT TERMS & CONDITIONS
       Go to: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means
       Scroll down and click "Download data" tab
       Accept the Copernicus license terms
    
    3. GET YOUR API KEY
       Go to: https://cds.climate.copernicus.eu/how-to-api
       Copy your UID and API Key
    
    4. INSTALL CDS API CLIENT
       Run in terminal:
       pip install cdsapi
    
    5. CREATE CONFIGURATION FILE
       Create a file called .cdsapirc in your home directory:
       
       Linux/Mac:
       ~/.cdsapirc
       
       Windows:
       C:\\Users\\<YourUsername>\\.cdsapirc
       
       File contents (replace with your actual credentials):
       url: https://cds.climate.copernicus.eu/api
       key: <UID>:<API-KEY>
       
       Example:
       url: https://cds.climate.copernicus.eu/api
       key: 12345:abcdef12-3456-7890-abcd-ef1234567890
    
    6. RUN THIS SCRIPT
       python download_era5_data.py
    
    ═══════════════════════════════════════════════════════════════════════
    
    For more help, see: https://cds.climate.copernicus.eu/how-to-api
    """)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--setup':
        setup_cds_api()
    elif len(sys.argv) > 1 and sys.argv[1] == '--verify':
        verify_downloads()
    else:
        # Run the download
        download_all_data()
        
        # Verify after download
        print("\n")
        verify_downloads()

