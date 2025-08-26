import xarray as xr
from gen_input import ADDITIONALVARS

def gen_ecrad_dset(model_fields: xr.Dataset, solar_irradiance: float,
                   cosine_sz_angle: xr.DataArray, aerosol_mmr: xr.Dataset,
                   ghg_data: xr.Dataset,
                   lut_dset: xr.Dataset = None, lut_recipes: dict = None,
                   rectangular_grid: bool = True, ignore_iseed: bool = True
                  ):
    import numpy as np
    import cf_xarray as cfxr
    import ifs_tools as ifst

    # ensure correct type!
    model_fields       = model_fields.astype("float32")
    cosine_sz_angle    = cosine_sz_angle.astype("float32")
    aerosol_mmr        = aerosol_mmr.astype("float32")
    ghg_data           = ghg_data.astype("float32")


    # Take a required 2D field as mold
    iseed = xr.full_like(model_fields["skt"], 1, dtype=np.int32).rename("iseed")
    if not ignore_iseed:
        print("Generating seeds...")
        iseed[...] = GlobalGenerator.float64(low=0, high=1.e10, size=iseed.shape)

    # ecRad ice content is sum of ice and snow water content
    q_ice = (model_fields["ciwc"] + model_fields["cswc"]).rename("q_ice")

    # Compute ml p and t
    print("Computing half-level p and t...")
    p_full = model_fields["p"].astype(np.float32).squeeze()
    p_half = ifst.compute_ml_pressure(model_fields, half_level=True).astype(np.float32).squeeze().compute()
    t_half = ifst.compute_hl_temperature(p_half=p_half, p_full=p_full,
                                    t_full=model_fields["t"], skt_sfc=model_fields["skt"]).astype(np.float32).compute()

    print("done!")

    # Compute liquid and ice effective radius in meters (requires p in model_fields)
    if lut_recipes and lut_dset:
        print("re_liquid from LUT!")
        llut = True
        import lut_tools as lutt
        from ifs_tools import RD
        dens_full = p_full/(RD*model_fields["t"].astype(np.float32))
        aero_mcon_for_lut = (dens_full*aerosol_mmr).sel(lev=[129,])
        with_extra_fields = "bin_num" in ADDITIONALVARS or "ccn_num" in ADDITIONALVARS or "ccn_act" in ADDITIONALVARS
        lut_species_mcon = lutt.compute_lut_species_from_ifs_species(lut_recipes, aero_mcon_for_lut)
        ccn_fields = lutt.compute_cdnc(lut_species_mcon, lut_dset,
                                      with_extra_fields=with_extra_fields).squeeze(drop=True)

        re_liquid = ifst.compute_liquid_reff_ifslut(dset=model_fields, cdnc_fields=ccn_fields["cdnc"]).compute()*1.e-6
        #re_liquid=xr.ones_like(p_full)
    else:
        llut = False
        w_10m = np.sqrt(model_fields["10u"]**2+model_fields["10v"]**2)
        ccn_fields = ifst.compute_ccn_ifs(ws=w_10m, lsm=model_fields["lsm"]).astype(np.float32)
        re_liquid = ifst.compute_liquid_reff_ifs(dset=model_fields, ccn_fields=ccn_fields).astype(np.float32).compute()*1.e-6

    # Ice particle size
    print("Computing re_ice...")
    re_ice    = ifst.compute_ice_reff_ifs(dset=model_fields, ).astype(np.float32).compute()*1.e-6
    print("done!")

    data_vars = {
        "solar_irradiance"       : xr.DataArray(np.float32(solar_irradiance)),
        "skin_temperature"       : model_fields["skt"],
        "cos_solar_zenith_angle" : cosine_sz_angle,
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
            continue
        match additional_var:
            case "lre":
                data_vars["lre"] = re_liquid
            case "ire":
                data_vars["ire"] = re_ice
            case "cdnc":
                data_vars["cdnc"] = ccn_fields["cdnc"] if llut else ccn_fields
            case "bin_num" if llut:
                data_vars["bin_num"] = xr.concat([ccn_fields[f"aero{i}_bin"].assign_coords(lutspec=i) for i in range(1,5)], dim="lutspec")
            case "ccn_num" if llut:
                data_vars["ccn_num"] = xr.concat([ccn_fields[f"aero{i}_ccn"].assign_coords(lutspec=i) for i in range(1,5)], dim="lutspec")
            case "ccn_act" if llut:
                data_vars["ccn_act"] = xr.concat([ccn_fields[f"aero{i}_act"].assign_coords(lutspec=i) for i in range(1,5)], dim="lutspec")
            case _:
                print(f"Warning in packer: Ignoring {additional_var} from ADDITIONALVARS")

    print("Merging ecrad_dset...")
    ecrad_dset = xr.merge(
        [val.rename(key) for key,val in data_vars.items()],
        compat="minimal"
    ).assign_attrs({"Aerosols map"   : aerosol_mmr.attrs["aero_map"],
                     "Aerosols types" : aerosol_mmr.attrs["aero_typ"]}).compute()
    if rectangular_grid:
        lonlats = ["lon", "lat"]
        #ecrad_dset = ecrad_dset.transpose("lon", "lat", ..., "half_level", "lev")
        ecrad_dset = ecrad_dset.stack(col=lonlats)
        ecrad_dset = cfxr.encode_multi_index_as_compress(ecrad_dset, "col")
        hordim="col"
    else:
        hordim = list(set(model_fields["p"].dims) - set(["lev"]))[0]
    print(f"Horizontal dimension name: {hordim}")
    ecrad_dset = ecrad_dset.transpose(hordim, ..., "half_level", "lev")
    print("done!")

    for var_to_drop in ("time", "fcdate", "levaux"):
        if var_to_drop in ecrad_dset:
            ecrad_dset = ecrad_dset.drop_vars(var_to_drop)

    return ecrad_dset
