from pathlib import Path
from collections import defaultdict
import re
import numpy as np
import xarray as xr
import pyswi
from pyswi.swi_ts import calc_swi_ts
import numpy as np
import xarray as xr
import zarr
from numpy.lib.recfunctions import unstructured_to_structured
import logging
import sys

def get_gain(ds, i, j):    
    vars = ['gain_nom', 'gain_denom', 'gain_last_jd', 'gain_noise']
    gain = {}
    if 'gain_nom' in ds.keys():   
        for var in vars:   
            gain[var] = ds[var][i,j]        
    else:
        gain = None
    return (gain)
    

def average_ssm_to_zarr(input_dir, output_zarr):
    # DIR_IN = Path("test_ssm/ssm_nc/E042N012_BCnotav/E042N012_basic")
    # output_zarr = 'test_ssm/ssm_zarr/ssm_summer_2026_av_basic.zarr'
    DIR_IN = Path(input_dir)

    date_re = re.compile(r"(\d{8})\d{6}_EU")

    chunk_size_time = 5
    chunk_size_x = 100
    chunk_size_y = 100

    # -----------------------------------------------------
    # Collect files by date
    # -----------------------------------------------------
    files_by_date = defaultdict(list)
    for f in DIR_IN.glob("*.nc"):
        m = date_re.search(f.name)
        if m:
            files_by_date[m.group(1)].append(f)

    n_timestamps = len(files_by_date)
    logging.info(f"Found {n_timestamps} dates to process")

    x = None
    y = None
    time_list = []
    ssm_list = []
    lc_list = []
    pf_list = []

    # -----------------------------------------------------
    # Process each date (average + time to 12:00)
    # -----------------------------------------------------
    for i, date in enumerate(sorted(files_by_date)):
        flist = sorted(files_by_date[date])

        template_ds = xr.open_dataset(flist[0])
        if x is None:
            x = template_ds['x'].values
            y = template_ds['y'].values

        sm_list = []
        for fn in flist:
            ds = xr.open_dataset(fn)
            sm = ds["surface_soil_moisture"][0, :, :]
            sm = sm.where(sm >= 0)
            sm = sm.where(sm != 655.35)
            sm_list.append(sm)
            ds.close()

        sm_mean = xr.concat(sm_list, dim="file").mean(dim="file", skipna=True)
        sm_mean = sm_mean.fillna(655.35)

        noon = f"{date[:4]}-{date[4:6]}-{date[6:]}T12:00:00"
        time_list.append(np.datetime64(noon))
        ssm_list.append(sm_mean.values)
        lc_list.append(template_ds['land_cover_flag'][0, :, :].values)
        pf_list.append(template_ds['processing_flag'][0, :, :].values)
        template_ds.close()

        if i % 50 == 0:
            logging.info(f"Processed date {i + 1}/{n_timestamps}")

    # -----------------------------------------------------
    # Build dataset and write to zarr
    # -----------------------------------------------------
    ds_out = xr.Dataset(
        data_vars={
            'surface_soil_moisture': (('x', 'y', 'time'), np.stack(ssm_list, axis=-1)),
            'land_cover_flag': (('x', 'y', 'time'), np.stack(lc_list, axis=-1)),
            'processing_flag': (('x', 'y', 'time'), np.stack(pf_list, axis=-1)),
        },
        coords={
            'x': x,
            'y': y,
            'time': np.asarray(time_list, dtype='datetime64[ns]'),
        },
        attrs={'Conventions': 'CF-1.8'}
    )

    logging.info("Writing to zarr...")
    encoding = {
        'surface_soil_moisture': {'chunks': (chunk_size_x, chunk_size_y, chunk_size_time), 'dtype': 'float32'},
        'land_cover_flag': {'chunks': (chunk_size_x, chunk_size_y, chunk_size_time), 'dtype': 'float32'},
        'processing_flag': {'chunks': (chunk_size_x, chunk_size_y, chunk_size_time), 'dtype': 'float32'},
        'x': {'chunks': (chunk_size_x,), 'dtype': 'float64'},
        'y': {'chunks': (chunk_size_y,), 'dtype': 'float64'},
    }

    ds_out.to_zarr(output_zarr, mode='w', encoding=encoding, zarr_version=2)

    logging.info("Consolidating metadata...")
    zarr.consolidate_metadata(output_zarr)

    logging.info(f"Successfully created {output_zarr}")

def pyswi_run(ssm_zarr, basic_zarr=None):
    # input dataset path, the dataset should contain the following variables: surface_soil_moisture, time, x, y 
    input_file = ssm_zarr #'test_ssm/ssm_zarr/ssm_summer_2026_BCav.zarr'

    # output dataset path, the output dataset will contain the following variable: swi_10, time, x, y 
    output_filename = ssm_zarr

    # reading ssm data from zarr
    zarr.consolidate_metadata(input_file)
    ds = xr.open_zarr(input_file, consolidated=True, decode_cf=True)

    #ssm_root = zarr.open(filename, mode='r')
    ssm_data = ds['surface_soil_moisture'].values
    time = ds['time'].values
    x_coords = ds['x'].values
    y_coords = ds['y'].values

    # converting time to julian dates
    time_ns = time.astype('datetime64[ns]').view(np.int64).astype(np.float64)
    juldates = time_ns / 86400000000000.0 + 2440587.5

    sort_idx = np.argsort(juldates)
    juldates_sorted = juldates[sort_idx]

    # defining the t_value for SWI calculation (it can be also array, for several values like [10, 20, 30])
    t_value = np.array([10], dtype=np.int32)

    nx, ny, nt = ssm_data.shape
    print(f"Processing {nx} x {ny} pixels with {nt} time steps")

    swi10_output = np.full((nx, ny, nt), np.nan, dtype=np.float32)

    # define the pixels, where at least one valid SSM value is present (SSM <= 100)
    valid_pixels = []
    for i in range(nx):
        for j in range(ny):
            if np.any(ssm_data[i, j, :] <= 100):
                valid_pixels.append((i, j))
    print(f"Found {len(valid_pixels)} valid pixels out of {nx*ny}")

    nom = np.full((nx, ny), np.nan, dtype=np.float32)
    denom = np.full((nx, ny), np.nan, dtype=np.float32)
    denom = np.full((nx, ny), np.nan, dtype=np.float32)
    last_jd = np.full((nx, ny), np.nan, dtype=np.float32)
    nom_noise = np.full((nx, ny), np.nan, dtype=np.float32)
    # process each pixel: time series of SSM values is extracted, 
    # sorted by time, and passed to the SWI calculation function
    for idx, (i, j) in enumerate(valid_pixels):
        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1}/{len(valid_pixels)} valid pixels")
        
        ssm = ssm_data[i, j, sort_idx].copy()
        ssm[ssm > 100] = np.nan
        juldates_pixel = juldates_sorted.copy()
        
        if np.all(np.isnan(ssm)):
            continue
        
        valid_mask = ~np.isnan(ssm)
        
        ssm_valid = ssm[valid_mask]
        juldates_valid = juldates_pixel[valid_mask]
        
        dtype = np.dtype([("sm_jd", np.float64), ("sm", np.float32)])
        ssm_ts = unstructured_to_structured(
            np.hstack((juldates_valid[:, np.newaxis], 
                    ssm_valid[:, np.newaxis])), 
            dtype=dtype
        )

        # calculate SWI time series for the pixel using pyswi package        
        gain_in = get_gain(ds, i, j)

        swi_result, gain_out = calc_swi_ts(
            ssm_ts=ssm_ts,
            swi_jd=juldates_valid,
            t_value=t_value,
            gain_in=None
        )

        # save gain to the array 
        nom[i,j] = gain_out['nom'][0]
        denom[i,j] = gain_out['denom'][0]
        last_jd[i,j] = gain_out['last_jd']
        nom_noise[i,j] = gain_out['nom_noise'][0]

        swi_full = np.full(len(juldates_pixel), np.nan, dtype=np.float32)
        swi_full[valid_mask] = swi_result['swi_10']
        
        swi10_output[i, j, :] = swi_full

    ds_out = xr.Dataset(
        data_vars={
            "swi_10": (("x", "y", "time"), swi10_output),
            "gain_nom":(("x", "y"), nom),
            "gain_denom":(("x", "y"), denom),
            "gain_last_jd":(("x", "y"), last_jd),
            "gain_noise":(("x", "y"), last_jd)
        },
        coords={
            "x": x_coords,
            "y": y_coords,
            "time": time
        },
    )

    ds_out.to_zarr(output_filename, mode="a", zarr_version=2)
    print(f"SWI10 added to {output_filename}")


def main():
    path_to_ssm_nc = '/home/mpanfilo/Documents/PROJECTS/A-DROP/ADO_NRT_DIREX/pyswi_tests/test_ssm/ssm_nc/E042N012_BCnotav/E042N012_add'
    output_zarr = '/home/mpanfilo/Documents/PROJECTS/A-DROP/ADO_NRT_DIREX/pyswi_tests/test_ssm/ssm_zarr/E042N012_add.zarr'
    #average_ssm_to_zarr(path_to_ssm_nc, output_zarr)

    ssm_zarr = '/home/mpanfilo/Documents/PROJECTS/A-DROP/ADO_NRT_DIREX/pyswi_tests/test_ssm/ssm_zarr/E042N012_basic.zarr'
    pyswi_run(ssm_zarr, basic_zarr=None)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logging.exception("Unhandled exception")
        sys.exit(1)
