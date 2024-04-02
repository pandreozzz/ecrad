# Setup
* Download era5 with needed parameters under new folder `./era5/data/` - example mars request listing model-level and surface parameters is under `./era5/mars_request.sh` - only rectangular grids supported for now
* Convert era5 fields to netcdf with cdo `cdo -f nc4 copy ${file} ${file/.grib/.nc}`
* Put the aerosol climatology files under `./clim_files` and set the variable `CAMSPATH` in `gen_input.py` (e.g. this one `https://sites.ecmwf.int/data/cams/aerosol_radiation_climatology/2003-2013/`)
* Compile the shared Fortran library under `./f_src` with `gfortran -fPIC -fopenmp -shared vertintp.f90 -o fvertintp.so`
* Set the global variable `INPUTSDIR` in `gen_input.py` and eventually create the directory, in this example is `./inputs`. The generated ecrad inputs will be output there. Set also the `EXPNAME` variable, so that the saved file will be `{EXPNAME}_{time}.nc`
* Launch `python gen_input.py --help` to see available options - Launch with something like: `python gen_input.py -i era5/data/*_1deg*.nc -t 1956-10-28T11:51 -a 3` to get fluxes from the avalable fields and simulating solar irradiance for the day 28-10-1956 at 11:51 UTC, using the aerosol version 3 (the 3D climatology). Most of the time is I/O operations on disk - so it could be worth creating `INPUTSDIR` in a fast temporary filesystem
* Have a look at the script outputs, it tells the mapping from the model fields times to the created ecrad input fils. Note down the number and the mapping of aerosols for the optica properties and fill the ecrad namelist variable `i_aerosol_type_map` as instructed (see next step)
* Take namelists from `../` and check the aerosol mapping as well as the aerosol optics file `aerosol_ifs_<.....>.nc`

# Run ecrad

# Postprocessing
ecRad wants one horizontal dimension for the input file. This is obtained by stacking the "lon" and "lat" dimensions for horizontal grids - this information can be retrieved from the input file to unstack the output file and get the ecrad output on the original grid of the model fields. This is using `cf_xarrays`:
```
import xarray as xr
import cf_xarray as cfxr
inp_off_ecrad_dset = xr.open_dataset("<path_to_inputs>/control_1956-10-28T11:51.nc")
out_off_ecrad_dset = xr.open_dataset("<path_to_outputs>/control_1956-10-28T11:51.nc")
out_off_ecrad_dset = out_off_erad_dset.rename(column="col").assign_coords(col=inp_off_ecrad_dset.col, lat=inp_off_ecrad_dset.lat, lon=inp_off_ecrad_dset.lon)
out_off_ecrad_dset = cfxr.decode_compress_to_multi_index(out_off_ecrad_dset, "col").unstack("col").transpose(...,"lat", "lon")
```
