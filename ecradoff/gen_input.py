import os
import argparse
import numpy as np
import xarray as xr

# Use Dask multithread/processing (does not work)
USEDASK = False
MULTIDASK = False
if USEDASK:
    try:
        NTHREADS=int(os.environ["OMP_NUM_THREADS"])
    except:
        NTHREADS=1
    from dask.distributed import Client
    client = Client(processes=MULTIDASK, threads_per_worker=NTHREADS, memory_limit="32G")
    print(client)
    from multiprocessing import freeze_support

CLIMDIR = "./clim_files"
# If the climatology fields are not as mmr, but as
# level-integrated mass concentrations
CLIMISMMR = True
GACC = 9.8

INPUTSDIR = "./inputs"
EXPNAME = "expname"

# Required fields
REQUIREDVARS = ["lnsp", "t", "q", "clwc", "cswc", "ciwc", "o3",
                "skt", "fal", "u10", "v10", "?"]


# Additional variables to be included in the produced fields (ignored by ecrad!)
ADDITIONALVARS = ["tcc", "cdnc", "bin_num", "ccn_act", "ccn_num", "mu0_spreader",
                  "cos_sensor_zenith_angle", "cos_solar_zenith_angle",
                  "sensor_azimuth_angle", "solar_azimuth_angle", "sea_fraction"]

DA_DTTYPE = "datetime64[D]"
NS_DTTYPE = "datetime64[ns]"
NS_TDTYPE = "timedelta64[ns]"

CAMSPATHS = {
    #3: os.path.join(CLIMDIR, "aerosol_cams_3d_climatology_2003-2013.nc"),
    3: os.path.join(CLIMDIR, "aerosol_cams_climatology_43r3_v2_3D_no_compression_classic.nc"),
    4: os.path.join(CLIMDIR, "aerosol_cams_climatology_49r2_1951-2019_4D.nc"),
    5: os.path.join(CLIMDIR, "aerosol_cams_climatology_49r2_1951-2019_4D.nc")
}

GHGFILE = os.path.join(CLIMDIR, "greenhouse_gas_timeseries_CMIP6_SSP370_CFC11equiv_47r1.nc")

# Automatically populated by get_model_fields()
TIMEDIM = None

def get_parser():
    parser = argparse.ArgumentParser(prog='Ecrad input generator', description="tbd",
                                     epilog="Something informative")
    parser.add_argument("-n", "--exp-name",
                        type=str, required=True,
                        help="Name of the experiment")

    parser.add_argument("-i", "--model-files",
                        type=str, nargs="+", required=True,
                        help="The IFS output fields to use for offline computations. " +\
                        "Typically 1 ml and 1 sfc file. Only times present in both " +\
                        "datasets are loaded."
                       )
    parser.add_argument("-t", "--times",
                        type=str, nargs="+", default=["all",],
                        help="Model time to use (can be repeated). "+\
                        "Format is YYYY-MM-HH:THH:MM or index starting from 0." +\
                        "Note that ecRad input at each timestep is stored in a separate file." +\
                        "Default is \"all\" - for all model timesteps"
                       )
    parser.add_argument("-r", "--rectangular-grid",
                        action="store_true",
                        help="By default expects unstructured input grid (with 1 horizontal dimension)."+\
                        "Select True for rectangular grids (lon,lat) with 2 horizontal dimensions." +\
                        "False is for reduced grids, ...    NOT YET FULLY IMPLEMENTED!!"
                       )
    parser.add_argument("-a", "--aerosol-version",
                        type=int, default=3,
                        choices=[3, 4, 5],
                        help="Version of aerosol fields to use." +\
                        "v3: CY43R3-CY49R1 Bozzo et al. 2020 3D climatology" +\
                        "v4: CY49R2 4D climatology" +\
                        "v5: CY49R1 prognostic - has hydrophilic dust and new PSD for sulfates"
                       )
    parser.add_argument("-lld", "--liquid-lut-dset",
                        type=str, default=None,
                        help="LUT dset to compute cdnc from aerosol fields")

    parser.add_argument("-llr", "--liquid-lut-recipes",
                        type=str, default=None,
                        help="LUT recipes to compute cdnc from aerosol fields")

    parser.add_argument("-zz", "--zamu0-cosine_sz_angle",
                        action="store_true",
                        help="If the cosine of solar zenith angle should go to zero " +\
                        "or to a finite value (zamu0 true, as in ecrad calls from IFS)"
                        )
    parser.add_argument("-si", "--spreader-interval",
                        type=str, default="0s",
                        help="Define a spreader (requires zamu0 true) to simulate " +\
                        "time interpolation of fluxes. Interval is e.g. `15m` for 15 minutes"
                        )

    parser.add_argument("-m", "--time-select-method",
                        type=str, default="nearest",
                        help="Method to use if the times indicated in the sequence are not in the field."+\
                        "Note that solar irradiances are computed always at exact times" +\
                        "ONLY 'nearest' is supported. Ignore this option."
                        )
    return parser

def gen_reduced_lon(ds : xr.Dataset or xr.DataArray, rpoint_coord : str = "reduced_points"):
    tmp_ds = ds.copy()

    newlon = np.array([])
    newlat = np.array([])
    for rpn, lat in zip(tmp_ds["reduced_points"].values, tmp_ds["lat"].values):
        newlon = np.concat([newlon, np.arange(0, 360, 360/rpn)], axis=0)
        newlat = np.concat([newlat, [lat]*rpn], axis=0)

    tmp_ds = tmp_ds.drop_vars("lat")
    tmp_ds = tmp_ds.assign_coords(
            lon = xr.DataArray(data=newlon, dims=["values"]),
            lat = xr.DataArray(data=newlat, dims=["values"])
            )

    return tmp_ds



def get_model_fields(model_files: list, intimes: list):
    renamedic = {
            "longitude" : "lon",
            "latitude" : "lat",
            "rgrid" : "values" # For cdo nc4
            }
   # Parse input file
    try:
        input_filepaths = [os.path.realpath(f) for f in model_files]
    except:
        input_filepaths = None
        raise ValueError(f"Could not parse input:\n{args.model_files}")

    for fpath in input_filepaths:
        if not os.path.exists(fpath):
            raise ValueError(f"File does not exist:\n{fpath}")

    if USEDASK:
        model_fields = xr.open_mfdataset(input_filepaths, join="inner",
                                         parallel=USEDASK)#.sel(lon=slice(0,2), lat=slice(0-2))
    else:
        model_fields = xr.merge([xr.load_dataset(p) for p in input_filepaths])

    if not USEDASK:
        model_fields = model_fields.compute()

    global TIMEDIM
    TIMEDIM = "valid_time" if "valid_time" in model_fields else "time"

    if intimes[0] == "all":
        print("Using all model times")
        times = model_fields[TIMEDIM].values
        intimes = list(range(0,len(times)))
        time_by_index = False
    else:
        # Parse times
        try:
            times = [int(tstr) for tstr in intimes]
            time_by_index = True
        except:
            try:
                times = [np.datetime64(tstr).astype("datetime64[ns]") for tstr in intimes]
                time_by_index = False
            except:
                times = None
                time_by_index = False
                raise ValueError(f"Could not parse times either as sequence of steps or as datatime strings:\n{args.time}")

    # Select model times
    if time_by_index:
        model_fields = model_fields.isel({TIMEDIM: times})
    else:
        model_fields = model_fields.sel({TIMEDIM: times}, method="nearest")


    # Store original model times and redefine those in model fields
    model_origtimes = model_fields[TIMEDIM]
    if not time_by_index:
        model_fields[TIMEDIM] = times

    print("Using following time mappings for physical fields:\n"+\
          "\n".join([f"{str(intime):14} -> {np.datetime_as_string(mtime)[:16]}"
                     for intime, mtime in zip(intimes, model_origtimes.values)]))

    for vset in (model_fields.variables, model_fields.dims):
        model_fields = model_fields.rename({var: renamedic[var.lower()]
                                            for var in vset if var in renamedic})

    if "lon" not in model_fields:
        model_fields = gen_reduced_lon(model_fields)

    return model_fields

def get_aerosol_clim(aerosol_version, model_times):
    import aeromaps
    from aeromaps import PDIM, HLDPDIM, HLPDIM

    model_dates = xr.DataArray(
        data=np.unique(model_times.dt.date.astype("datetime64[ns]")),
        dims=TIMEDIM)

    cams_dset = xr.load_dataset(CAMSPATHS[aerosol_version])

    # Fix this with correct epoch choice
    if aerosol_version > 3:
        cams_dset = cams_dset.sel(epoch=2015)

    cams_tintp = aeromaps.interpolate_monthly_aerosols(cams_dset, model_dates)

    if PDIM not in cams_dset:
        if (HLPDIM in cams_tintp) and (HLDPDIM in cams_tintp):
            cams_tintp[PDIM] = cams_tintp[HLPDIM] - cams_tintp[HLDPDIM]/2
        else:
            raise ValueError("Error while reading aerosol climatology:"+\
                    f"Could not find {PDIM} and {HLPDIM} and {HLDPDIM} not found.")
    if not CLIMISMMR:
        aero_conversion = GACC/cams_tintp[HLDPDIM]
    else:
        aero_conversion = 1.

    aero_ecrad, aero_typ, aero_map = aeromaps.gen_aerosol_dataset(cams_dset=cams_tintp, aero_version=aerosol_version)

    aero_type_coord = "aer_type"
    aerosol_mmr = xr.concat(
        [aero_conversion*aero_ecrad[aero].expand_dims(
            dim=aero_type_coord, axis=0
            ).assign_attrs(unit="Kg/Kg", long_name=aero.replace("_", " ")+" mass mixing ratio")
         for aero in aero_typ],
        dim=aero_type_coord
    ).rename("aerosol_mmr").assign_coords({aero_type_coord:aero_typ})


    aero_map_str = ", ".join([f"{idx:d}" for idx in aero_map])
    aerosol_mmr = aerosol_mmr.assign_attrs(
        aero_map=aero_map_str,
        aero_typ=", ".join(aero_typ))


    print(f"\n\nMapping of aerosol optical properties (version {aerosol_version}):" +\
          "\n-------------------\n" +\
          "\n".join([f"{typ:30} -> {idx:3d}" for typ, idx in zip(aero_typ, aero_map)]) +\
          "\n-------------------\n" +\
          "Use following ordered map in ecrad namelist:\n" +\
          f"i_aerosol_type_map = {aero_map_str}\n\n"
         )
    return xr.merge([cams_tintp[PDIM], aerosol_mmr])

def get_ghg_data(model_times):

    def preprocess_ghg_dset(ds):
        ds = ds.sel(time=slice(1768, 2262))
        years = np.floor(ds["time"].values).astype(int)
        fracs = ds["time"].values - years

        ds = ds.drop_vars("time").rename(time=TIMEDIM)
        one_year = np.timedelta64(1, 'Y').astype(NS_TDTYPE)
        ds = ds.assign_coords({
            TIMEDIM: xr.DataArray(
            data=[np.datetime64(f"{y:4d}-01-01").astype(NS_DTTYPE)+one_year*f
                  for y, f in zip(years, fracs)],
            dims=[TIMEDIM,])})
        return ds

    model_dates = xr.DataArray(
        data=np.unique(model_times.dt.date.astype("datetime64[ns]")),
        dims=TIMEDIM)

    # Time should be from fcdate
    ghg_data = preprocess_ghg_dset(xr.open_dataset(
        GHGFILE,
        decode_times=False))


    ghg_data = ghg_data.interp(**{
        TIMEDIM: model_dates, "method":"linear",
        "kwargs": {"fill_value": "extrapolate"}}).astype("float32").load()

    return ghg_data


def get_args(model_fields: xr.Dataset, aerosol_fields: xr.Dataset,
             ghg_data: xr.Dataset, lut_dset: xr.Dataset, lut_recipes: xr.Dataset,
             zamu0: bool, spreader_deltas: np.ndarray, time: xr.DataArray,
             rectangular_grid: bool):
    import minieot
    import ifs_tools
    import aeromaps
    this_date = time.dt.date.values.astype(NS_DTTYPE)

    # Get the model fields
    this_model = model_fields.sel({TIMEDIM: time}).squeeze()

    # Get the aerosol_mmr and pressure fields
    this_aero_fields = aerosol_fields.sel({TIMEDIM: this_date}).squeeze()
    this_aero_fields[TIMEDIM] = time
    this_model["p"] = ifs_tools.compute_ml_pressure(this_model, half_level=False).squeeze()
    this_aero_fields_intp = aeromaps.interpolate_3d_aerosols(this_aero_fields, this_model["p"])
    this_aero_fields_intp = this_aero_fields_intp.rename("aerosol_mmr").assign_attrs(this_aero_fields["aerosol_mmr"].attrs)

    # Get the CDNC values from LUT
    if lut_recipes and lut_dset:
        import lut_tools as lutt
        lut_recipes_dict = lutt.get_dics_from_recipes_dset(lut_recipes)
    else:
        lut_dset = None
        lut_recipes_dict = None

    # Get the ghg data
    this_ghg_data = ghg_data.sel({TIMEDIM: this_date}).squeeze()
    this_ghg_data[TIMEDIM] = time

    # Irradiance
    ThisIrradiance = minieot.Irradiance(time.values)
    solar_irradiance = ThisIrradiance.solar_irr
    cosine_sz_angle = ThisIrradiance.mu0_cos_sza_deg(phi=model_fields.lat,
                                                     lam=model_fields.lon, zamu0=zamu0)
    # generate spreader?
#    this_mu0_spreader = 1.
#    if spreader_deltas:
#        for spr_step in spreader_deltas*time:
#            tmpirr = minieot.Irradiance(spr_step.values)
#            this_mu0_spreader = this_mu0_spreader + tmpirr.mu0_cos_sza_deg(phi=model_fields.lat,
#                                                                           lam=model_fields.lon, zamu0=False)
#        this_mu0_spreader = this_mu0_spreader/(len(spreader_deltas)*cosine_sz_angle)


    return dict(
        model_fields=this_model,
        solar_irradiance=solar_irradiance,
        cosine_sz_angle=cosine_sz_angle,
        aerosol_mmr=this_aero_fields_intp,
        ghg_data=this_ghg_data,
        lut_dset=lut_dset,
        lut_recipes=lut_recipes_dict,
        rectangular_grid=rectangular_grid
        #cosine_sza_spreader=this_mu0_spreader
        #cdnc_fields=arg_cdnc,
        #descr=f"yo_{np.datetime_as_stringtime)[:16]}"
        #rectangular_grid=args....
    )

def driver():
    import packer

    ###
    # Parse arguments
    ###
    parser = get_parser()

    args = parser.parse_args()
    EXPNAME = args.exp_name
    print(f"Experiment name: {EXPNAME}")

    ###
    # Model fields
    ###
    print("Loading model fields...")
    model_fields = get_model_fields(args.model_files, args.times)

    # Extract model times
    print("Extracting model times...")
    model_times = model_fields[TIMEDIM]

    ###
    # Organize aerosol fields
    ###
    print("Organizing aerosol fields...")
    aerosol_fields = get_aerosol_clim(args.aerosol_version, model_times)


    ###
    # LUT for liquid CDNC?
    ###
    lut_dset, lut_recipes = args.liquid_lut_dset, args.liquid_lut_recipes
    if lut_recipes and lut_dset:
        print("Using LUT for CDNC")
        lut_recipes = xr.open_dataset(lut_recipes)
        lut_dset = xr.open_dataset(lut_dset)
    elif lut_recipes or lut_dset:
        print("Warning: Both LUT recipes and dset are needed to compute CDNC from LUT!" +\
              "All LUT settings will be ignored.")

    ###
    # Fetch GHG concentrations
    ###
    print("Fetching GHG data...")
    ghg_data = get_ghg_data(model_times)

    ###
    # Spreader Interval
    ###

    spreader_deltas = None
#    try:
#        spr_mm = re.match("(\d+)(\D+)", args.spreader_interval)
#        spread_interval = np.timedelta64(spr_mm.group(1), spr_mm.group(2))
#    except:
#        spread_interval = np.timedelta64(0,'s')
#    gen_spreader = spread_interval > np.timedelta64(0,'s') and args.zamu0_cosine_sz_angle
#    time_diff = np.unique(model_times.diff(dim="time"))
#    if len(time_diff) == 1:
#        time_diff = time_diff.values.item()
#        nints = int(time_diff/spread_interval)
#        if time_diff =  nints*spread_interval:
#            spreader_deltas = np.arange(-int(nints/2), nints-int(nints/2))*spread_interval

    ###
    # Each timestep is a separate ecrad input!
    ###
    arglist = []
    for this_time in model_times:
        arglist.append(get_args(model_fields, aerosol_fields, ghg_data,
                                lut_dset, lut_recipes, args.zamu0_cosine_sz_angle,
                                spreader_deltas, this_time, args.rectangular_grid))

    for args, time in zip(arglist, model_times):
        fpath = os.path.join(INPUTSDIR, f"{EXPNAME}_{np.datetime_as_string(time)[:16]}.nc")
        ecrad_dset = packer.gen_ecrad_dset(**args)
        print(f"Saving to {fpath}...")
        ecrad_dset.to_netcdf(fpath)


if __name__ == '__main__':
    if USEDASK:
        freeze_support()
    driver()
