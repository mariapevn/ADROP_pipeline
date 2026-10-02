# The tool to visualize the time series of SSM and SWI datacubes.
# Run the script and click with the left mouse button on the image.
# The time series of the clicked pixel will be displayed in the right panel.
# Can be adapted to visualize other datacubes, just change the input Zarr files and the variable names.

# Create the conda environment from the provided file:
#   1. conda env create -f environment.yml
#   2. conda activate swi_analysis


import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import numpy as np
import zarr
from datetime import datetime


# Read the Zarr datasets
filename = '/home/mpanfilo/Documents/PROJECTS/A-DROP/ADO_NRT_DIREX/pyswi_tests/test_ssm/ssm_zarr/ssm_summer_2026_BCav.zarr'
zarr.consolidate_metadata(filename)
ds_ssm_av = xr.open_zarr(filename, consolidated=True)

filename = '/home/mpanfilo/Documents/PROJECTS/A-DROP/ADO_NRT_DIREX/pyswi_tests/test_ssm/ssm_zarr/E042N012_add.zarr'
zarr.consolidate_metadata(filename)
ds_ssm_notav = xr.open_zarr(filename, consolidated=True)

filename = '/home/mpanfilo/Documents/PROJECTS/A-DROP/ADO_NRT_DIREX/pyswi_tests/test_ssm/swi_pyswi_zarr/swi10_summer_2026_pyswi_BCav.zarr'
zarr.consolidate_metadata(filename)
ds_swi_pyswi = xr.open_zarr(filename, consolidated=True)

filename = '/home/mpanfilo/Documents/PROJECTS/A-DROP/ADO_NRT_DIREX/pyswi_tests/test_ssm/ssm_zarr/E042N012_basic.zarr'
zarr.consolidate_metadata(filename)
ds_swi_sam = xr.open_zarr(filename, consolidated=True)

# in the dataset swi from Samuel code the 1st 8 timestemps are missing, therefore we skip them in the others
ssm_av = ds_ssm_av['surface_soil_moisture'].values
time_av_ssm = ds_ssm_av['time'].values

ssm = ds_ssm_notav['swi_10'].values
ssm[ssm > 100] = np.nan
time_ssm = ds_ssm_notav['time'].values

swi_pyswi = ds_swi_pyswi['swi_10'].values
time_pyswi = ds_swi_pyswi['time'].values

swi_sam = ds_swi_sam['swi_10'].values
time_sam = ds_swi_sam['time'].values

marker = None

fig, (ax_img, ax_ts) = plt.subplots(1, 2, figsize=(10, 5))
ax_ts.clear()
ax_ts.set_xlabel("Time")
ax_ts.set_ylabel("Value")

def find_nearest_valid_pixel(x_click, y_click, valid_mask):
    """Return (x, y) of nearest valid pixel."""
    yy, xx = np.where(valid_mask)

    if len(xx) == 0:
        return None

    dist2 = (xx - x_click)**2 + (yy - y_click)**2
    idx = np.argmin(dist2)

    return xx[idx], yy[idx]


def main():
    time = time_pyswi
    time_sorted_idx = np.argsort(time)
    time = time[time_sorted_idx]
    swi10 = swi_pyswi[:, :, time_sorted_idx]

    nx, ny, nt = swi10.shape
    print(f"Data shape: {nx} x {ny} x {nt}")

    swi_latest = np.full((nx, ny), np.nan)

    t = time[-1]
    i = len(time) - 1

    time_ns = t.astype('datetime64[ns]')
    dt = time_ns.astype('datetime64[s]').astype('datetime64[D]').astype(datetime)
    date_str = dt.strftime('%Y%m%d-%H:%M:%S')
    
    swi_latest = swi10[:, :, i]
    
    # ----------- plot the map
    
    im = ax_img.imshow(swi_latest, cmap='RdBu', vmin=0, vmax=100)   

    cbar = plt.colorbar(
        im,
        ax=ax_img,
        orientation='horizontal',
        fraction=0.05,
        pad=0.05,
        aspect=40
    )
    ax_img.set_title(f"SWI_10 - {date_str}")
    ax_img.axis('off')

    def onclick(event):
        ax_ts.set_title("Click a pixel")
        ax_ts.clear()
        ax_ts.set_xlabel("Time")
        ax_ts.set_ylabel("Value")
        global marker

        if event.inaxes != ax_img:
            return

        if event.button != MouseButton.LEFT:
            return

        if event.xdata is None or event.ydata is None:
            return

        x_click = int(round(event.xdata))
        y_click = int(round(event.ydata))

        pixel = find_nearest_valid_pixel(x_click, y_click, ~np.isnan(swi_latest))
        
        if pixel is None:
            print("No valid pixels found.")
            return

        x, y = pixel

        print(f"Selected pixel: ({x}, {y})")

        # Remove previous marker
        if marker is not None:
            marker.remove()

        # Draw new marker
        marker, = ax_img.plot(
            x, y,
            marker='*',
            color='k',
            markersize=18,
            markeredgecolor='white',
            markeredgewidth=1.5
        )

        # swap the coordinates
        x,y=y,x
        # Get time series data       

        ax_ts.plot(time_av_ssm, ssm_av[x, y], 'r-o', label='SSM averaged')
        ax_ts.plot(time_pyswi, swi_pyswi[x,y], 'm-o', label='SWI pyswi, entire array')
        ax_ts.plot(time_sam, swi_sam[x,y], 'b-.', label='SWI basic')
        ax_ts.plot(time_ssm, ssm[x,y],'g-.', label='add SWI, gain out from basic')
        
        ax_ts.tick_params(axis='x', rotation=45)
        ax_ts.legend()
        ax_ts.set_title(f'SSM and SWI time series for pixel ({y},{x}) (%)')
        
        ax_ts.set_title(f"Pixel (x={y}, y={x})")

        ax_ts.set_ylim([0, 100])
        
        # Update time series plot
        ax_ts.relim()
        ax_ts.autoscale_view()
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

if __name__ == "__main__":    
    main()
    
