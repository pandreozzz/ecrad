import xarray as xr
ADDITIONALVARS = []
RECTGRID = True
GHGFILE = "./clim_files/greenhouse_gas_timeseries_CMIP6_SSP370_CFC11equiv_47r1.nc"

def gen_ecrad_dset(model_fields : xr.Dataset, solar_irradiance : float,
                   cosine_sz_angle : xr.DataArray, aerosol_mmr : xr.Dataset,
                   #lre_fields : xr.DataArray, ire_fields : xr.DataArray,
                   ignore_iseed : bool = True
                  ):
    import numpy as np
    import cf_xarray as cfxr
    import ifs_tools as ifst

    # ensure correct type!
    model_fields       = model_fields.astype("float32")
    cosine_sz_angle    = cosine_sz_angle.astype("float32")
    aerosol_mmr        = aerosol_mmr.astype("float32")

    # Time should be from fcdate
    ghg_data = xr.open_dataset(
        GHGFILE,
        decode_times=False).sel(time=2020, method="nearest").squeeze().astype("float32")

    # Take a required 2D field as mold
    iseed = xr.full_like(model_fields["skt"], 1, dtype=np.int32).rename("iseed")
    if not ignore_iseed:
        print("Generating seeds...")
        iseed[...] = GlobalGenerator.float64(low=0, high=1.e10, size=iseed.shape)

    # ecRad ice content is sum of ice and snow water content
    q_ice = (model_fields["ciwc"] + model_fields["cswc"]).rename("q_ice")

    # Compute ml p and t
    p_full = model_fields["p"].astype(np.float32).squeeze()
    p_half = ifst.compute_ml_pressure(model_fields, half_level=True).astype(np.float32).squeeze().compute()
    t_half = ifst.compute_hl_temperature(p_half=p_half, p_full=p_full,
                                    t_full=model_fields["t"], skt_sfc=model_fields["skt"]).astype(np.float32).compute()

    # Compute liquid and ice effective radius in meters (requires p in model_fields)
    w_10m = np.sqrt(model_fields["10u"]**2+model_fields["10v"]**2)
    ccn_fields = ifst.compute_ccn_ifs(ws=w_10m, lsm=model_fields["lsm"]).astype(np.float32).compute()
    re_liquid = ifst.compute_liquid_reff_ifs(dset=model_fields, ccn_fields=ccn_fields).astype(np.float32).compute()*1.e-6
    re_ice    = ifst.compute_ice_reff_ifs(dset=model_fields, ).astype(np.float32).compute()*1.e-6


    data_vars = {
        "solar_irradiance"       : xr.DataArray(np.float32(solar_irradiance)),
        "skin_temperature"       : model_fields["skt"],
        "cos_solar_zenith_angle" : cosine_sz_angle, # approx val
        "sw_albedo"              : model_fields["fal"], # use spectrally constant albedo
        "lw_emissivity"          : xr.full_like(model_fields["fal"], 0.99, dtype="float32"),
        "iseed"                  : iseed,
        "pressure_hl"            : p_half,
        "temperature_hl"         : t_half,
        "h2o_mmr"                : model_fields["q"],
        "o3_mmr"                 : model_fields["o3"],
        "o2_vmr"                 : xr.DataArray(np.float32(0.20944)),
        #"co_vmr"                 : xr.DataArray(np.float32(1.e-6)),
        "hcfc22_vmr"             : xr.DataArray(np.float32(0.)), #xr.DataArray(np.float32(240*1.e-12)),
        "ccl4_vmr"               : xr.DataArray(np.float32(0.)), #xr.DataArray(np.float32(77*1.e-12)),
        #"no2_vmr"                : xr.DataArray(np.float32(1.e-8)),
        "co2_vmr"                : ghg_data["co2_vmr"],
        "ch4_vmr"                : ghg_data["ch4_vmr"],
        "n2o_vmr"                : ghg_data["n2o_vmr"],
        "cfc11_vmr"              : ghg_data["cfc11_vmr"],
        "cfc12_vmr"              : ghg_data["cfc12_vmr"],
        "aerosol_mmr"            : aerosol_mmr.compute(),
        "q_liquid"               : model_fields["clwc"],
        "q_ice"                  : q_ice,
        "re_liquid"              : re_liquid,
        "re_ice"                 : re_ice,
        "cloud_fraction"         : model_fields["cc"],
    }

    for additional_var in ADDITIONALVARS:
        if additional_var in model_fields.variables:
            data_vars = {**data_vars,**{additional_var : model_fields[additional_var]}}


    ecrad_dset = xr.merge(
        [val.rename(key) for key,val in data_vars.items()],
        compat="minimal"
    ).assign_attrs({"Aerosols map"   : aerosol_mmr.attrs["aero_map"],
                     "Aerosols types" : aerosol_mmr.attrs["aero_typ"]}).compute()
    if RECTGRID:
        lonlats = ["lon", "lat"]
        #ecrad_dset = ecrad_dset.transpose("lon", "lat", ..., "half_level", "lev")
        ecrad_dset = ecrad_dset.stack(col=lonlats)
        ecrad_dset = cfxr.encode_multi_index_as_compress(ecrad_dset, "col")
        hordim="col"
    else:
        hordim = list(set(model_fields.dims) - set(["lev"]))[0]
    ecrad_dset = ecrad_dset.transpose(hordim, ..., "half_level", "lev")

    for var_to_drop in ("time", "fcdate", "levaux"):
        if var_to_drop in ecrad_dset:
            ecrad_dset = ecrad_dset.drop_vars(var_to_drop)

    return ecrad_dset
