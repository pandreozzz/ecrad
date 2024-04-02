import os
import argparse
import numpy as np
import xarray as xr

# Use Dask multiprocessing (does not work)
USEDASK=False
MULTIDASK=True
if USEDASK:
    from dask.distributed import Client
    client = Client(processes=MULTIDASK)
    print(client)
    from multiprocessing import freeze_support

CLIMDIR="./clim_files"
# If the climatology fields are not as mmr, but as
# level-integrated mass concentrations
CLIMISMMR=False
GACC=9.8

INPUTSDIR = "./inputs"
EXPNAME   = "control"

# Additional variables to be included in the produced fields (ignored by ecrad!)
ADDITIONALVARS = ["tcc",]

DA_DTTYPE = "datetime64[D]"
NS_DTTYPE = "datetime64[ns]"

CAMSPATHS = {
    3 : os.path.join(CLIMDIR, "aerosol_cams_3d_climatology_2003-2013.nc"),
    #3 : os.path.join(CLIMDIR, "aerosol_cams_climatology_43r3_v2_3D_no_compression_classic.nc"),
    4 : os.path.join(CLIMDIR, "aerosol_cams_climatology_49r2_1951-2019_4D.nc"),
    5 : os.path.join(CLIMDIR, "aerosol_cams_climatology_49r2_1951-2019_4D.nc")
}
def get_parser():
    parser = argparse.ArgumentParser(prog='Ecrad input generator', description="tbd", epilog="Something informative")
    parser.add_argument("-i", "--model-files",
                        type=str, nargs="+", required=True,
                        help="The IFS output fields to use for offline computations. " +\
                        "Typically 1 ml and 1 sfc file. Only times present in both datasets are loaded."
                       )
    parser.add_argument("-t", "--times",
                        type=str, nargs="+", default=["0",],
                        help="Model time to use (can be repeated). "+\
                        "Format is YYYY-MM-HH:THH:MM or index starting from 0." +\
                        "Note that ecRad input at each timestep is stored in a separate file.",
                       )
    parser.add_argument("-r", "--reduced-grid",
                        action="store_true",
                        help="By default expects unstructured input grid (with 1 horizontal dimension)."+\
                        "Select True for rectangular grids (lon,lat) with 2 horizontal dimensions." +\
                        "False is for reduced grids, ... NOT YET FULLY IMPLEMENTED!!"
                       )
    parser.add_argument("-a", "--aerosol-version",
                        type=int, default=3,
                        choices=[3, 4, 5],
                        help="Version of aerosol fields to use." +\
                        "v3: CY43R3-CY49R1 Bozzo et al. 2020 3D climatology" +\
                        "v4: CY49R2 4D climatology" +\
                        "v5: CY49R2 4D climatology with hydrophilic dust"
                       )
    parser.add_argument("-m", "--time-select-method",
                        type=str, default="nearest",
                        help="Method to use if the times indicated in the sequence are not in the field."+\
                        "Note that solar irradiances are computed always at exact times" +\
                        "ONLY 'nearest' is supported. Ignore this option."
                        )
    return parser

def get_model_fields(model_files : list, intimes : list):
   # Parse input file
    try:
        input_filepaths = [os.path.realpath(f) for f in model_files]
    except:
        input_filepaths = None
        raise ValueError(f"Could not parse input:\n{args.model_files}")

    for fpath in input_filepaths:
        if not os.path.exists(fpath):
            raise ValueError(f"File does not exist:\n{fpath}")

    model_fields = xr.open_mfdataset(input_filepaths, join="inner", parallel=USEDASK)

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
        model_fields = model_fields.isel(time=times)
    else:
        model_fields = model_fields.sel(time=times, method="nearest")

    # Store original model times and redefine those in model fields
    model_origtimes = model_fields["time"]
    if not time_by_index:
        model_fields["time"] = times

    print("Using following time mappings for physical fields:\n"+\
          "\n".join([f"{str(intime):14} -> {np.datetime_as_string(mtime)[:16]}"
                     for intime,mtime in zip(intimes,model_origtimes.values)]))

    return model_fields

def get_aerosol_clim(aerosol_version, model_times):
    import aeromaps
    from aeromaps import PDIM, HLDPDIM, HLPDIM

    model_dates = xr.DataArray(
        data=np.unique(model_times.dt.date.astype("datetime64[ns]")),
        dims="time")

    cams_dset = xr.open_mfdataset(CAMSPATHS[aerosol_version], parallel=USEDASK)
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
            ).assign_attrs(unit="Kg/Kg", long_name=aero.replace("_"," ")+" mass mixing ratio")
         for aero in aero_typ],
        dim=aero_type_coord
    ).rename("aerosol_mmr").assign_coords({aero_type_coord:aero_typ})
    aero_map_str = ", ".join([f"{idx:d}" for idx in aero_map])
    aero_typ_str = ", ".join(aero_typ)
    aerosol_mmr  = aerosol_mmr.assign_attrs(aero_map=aero_map_str, aero_typ=aero_typ_str)
    aerosol_fields = xr.merge([cams_tintp[PDIM], aerosol_mmr])

    print(aerosol_mmr)
    exit

    print(f"\n\nMapping of aerosol optical properties (version {aerosol_version}):" +\
          "\n-------------------\n" +\
          "\n".join([f"{typ:30} -> {i:3d}" for typ,i in zip(aero_typ, aero_map)]) +\
          "\n-------------------\n" +\
          "Use following ordered map in ecrad namelist:\n" +\
          f"i_aerosol_type_map = {aero_map_str}\n\n"
         )
    return aerosol_fields

def get_args(model_fields : xr.Dataset, aerosol_fields : xr.Dataset, time : xr.DataArray):
    import minieot
    import ifs_tools
    import aeromaps
    this_date = time.dt.date.astype(NS_DTTYPE)

    # Get the model fields
    this_model = model_fields.sel(time=time).squeeze()

    # Get the aerosol_mmr and pressure fields
    this_aero_fields = aerosol_fields.sel(time=this_date).squeeze()
    this_aero_fields["time"] = time
    this_model["p"] = ifs_tools.compute_ml_pressure(this_model, half_level=False).squeeze()
    this_aero_fields_intp = aeromaps.interpolate_3d_aerosols(this_aero_fields, this_model["p"])
    this_aero_fields_intp = this_aero_fields_intp.rename("aerosol_mmr").assign_attrs(this_aero_fields["aerosol_mmr"].attrs)

    ThisIrradiance   = minieot.Irradiance(time.values)
    solar_irradiance = ThisIrradiance.solar_irr
    cosine_sz_angle  = ThisIrradiance.mu0_cos_sza_deg(phi=model_fields.lat,
                                                      lam=model_fields.lon, zamu0 = False)

    return dict(
        model_fields=this_model,
        solar_irradiance=solar_irradiance,
        cosine_sz_angle=cosine_sz_angle,
        aerosol_mmr=this_aero_fields_intp,
        #cdnc_fields=arg_cdnc,
        #descr=f"yo_{np.datetime_as_stringtime)[:16]}"
    )

def driver():
    import packer

    ###
    # Parse arguments
    ###
    parser = get_parser()
    # example test call
    #args = parser.parse_args("-i ../Ecradoff/rf13_ml_01deg.nc ../Ecradoff/rf13_sfc_01deg.nc -t 2018-02-20T04:20 2018-02-20T05:15 -m linear -r ".split())
    args = parser.parse_args()

    ###
    # Model fields
    ###
    model_fields = get_model_fields(args.model_files, args.times)

    # Extract model times
    model_times = model_fields["time"]

    ###
    # Organize aerosol fields
    ###
    aerosol_fields = get_aerosol_clim(args.aerosol_version, model_times)

    ###
    # Each timestep is a separate ecrad input!
    ###
    arglist = []
    for this_time in model_times:
        arglist.append(get_args(model_fields, aerosol_fields, this_time))

    for args,time in zip(arglist,model_times):
        fpath = os.path.join(INPUTSDIR, f"{EXPNAME}_{np.datetime_as_string(time)[:16]}.nc")
        ecrad_dset = packer.gen_ecrad_dset(**args)
        print(f"Saving to {fpath}...")
        ecrad_dset.to_netcdf(fpath)


if __name__ == '__main__':
    if USEDASK:
        freeze_support()
    driver()

