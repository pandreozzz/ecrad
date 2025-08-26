from dataclasses import dataclass
import numpy as np
import xarray as xr

#############################
# CAMS SPECIES NAMES
#############################
@dataclass
class ClimSpecies:
    longname: str
    shortname: str
    spectype: str
    spechydro: bool
    specbin: int
    def __post_init__(self):
        for (name, field_type) in self.__annotations__.items():
            if not isinstance(self.__dict__[name], field_type):
                current_type = type(self.__dict__[name])
                raise TypeError(f"The field `{name}` was assigned by `{current_type}`" +\
                        "instead of `{field_type}`")

AEROCAMSBUCKET = {
    "Sea_Salt_bin1"                  : ClimSpecies("Sea_Salt_bin1",
                                                   "SS1", "SS", True, 1),
    "Sea_Salt_bin2"                  : ClimSpecies("Sea_Salt_bin2",
                                                   "SS2", "SS", True, 2),
    "Sea_Salt_bin3"                  : ClimSpecies("Sea_Salt_bin3",
                                                   "SS3", "SS", True, 3),
    "Mineral_Dust_bin1"              : ClimSpecies("Mineral_Dust_bin1",
                                                   "DD1", "DD", False, 1),
    "Mineral_Dust_bin2"              : ClimSpecies("Mineral_Dust_bin2",
                                                   "DD2", "DD", False, 2),
    "Mineral_Dust_bin3"              : ClimSpecies("Mineral_Dust_bin3",
                                                   "DD3", "DD", False, 3),
    "Organic_Matter_hydrophilic"     : ClimSpecies("Organic_Matter_hydrophilic",
                                                   "OMH", "OM", True, 0),
    "Organic_Matter_hydrophobic"     : ClimSpecies("Organic_Matter_hydrophobic",
                                                   "OMN", "OM", False, 0),
    "Black_Carbon_hydrophilic"       : ClimSpecies("Black_Carbon_hydrophilic",
                                                   "BCH", "BC", True, 0),
    "Black_Carbon_hydrophobic"       : ClimSpecies("Black_Carbon_hydrophobic",
                                                   "BCN", "BC", False, 0),
    "Sulfates"                       : ClimSpecies("Sulfates",
                                                   "SU", "SU", True, 0),
    "Nitrate_fine"                   : ClimSpecies("Nitrate_fine",
                                                   "NI1", "NI", True, 1),
    "Nitrate_coarse"                 : ClimSpecies("Nitrate_coarse",
                                                   "NI2", "NI", True, 2),
    "Ammonium"                       : ClimSpecies("Ammonium",
                                                   "AM", "AM", True, 0),
    "Biogenic_Secondary_Organic"     : ClimSpecies("Biogenic_Secondary_Organic",
                                                   "BSO", "OB", True, 0),
    "Anthropogenic_Secondary_Organic": ClimSpecies("Anthropogenic_Secondary_Organic",
                                                   "ASO", "OA", True, 0),
    "Stratospheric_Sulfate"          : ClimSpecies("Stratospheric_Sulfate",
                                                   "SSU", "SSU", False, 0),
}

AEROPTICSFILE = "../data/aerosol_ifs_49R1_20230119.nc"

PHILIC_DESCR = \
"""
1: Sea salt, bin 1, 0.03-0.5 micron, OPAC
2: Sea salt, bin 2, 0.50-5.0 micron, OPAC
3: Sea salt, bin 3, 5.0-20.0 micron, OPAC
4: Hydrophilic organic matter, OPAC
5: Ammonium sulfate (for sulfate), GACP Lacis et al https://gacp.giss.nasa.gov/data_sets/
6: Secondary organic aerosol - biogenic, Moise et al 2015
7: Secondary organic aerosol - anthropogenic, Moise et al 2015
8: Fine mode Ammonium sulfate (for ammonia), GACP Lacis et al https://gacp.giss.nasa.gov/data_sets/
9: Fine mode Nitrate, GLOMAP
10: Coarse mode Nitrate, GLOMAP
11: Hydrophilic organic matter, Brown et al 2018
12: Sulfate, GACP Lacis et al https://gacp.giss.nasa.gov/data_sets/
13: Sulfate, GACP Lacis et al https://gacp.giss.nasa.gov/data_sets/ with modified size distribution
14: Desert dust, bin 1, 0.03-0.55 micron, Composite-Philic Non-Sphere-Scaling-Kandler (Balkanski et 2007 , Di Baggio 2017, Ryder et al 2019)
15: Desert dust, bin 2, 0.55-0.90 micron, Composite-Philic Non-Sphere-Scaling-Kandler (Balkanski el 2007 , Di Baggio 2017, Ryder et al 2019)
16: Desert dust, bin 3, 0.90-20.0 micron, Composite-Philic Non-Sphere-Scaling-Kandler (Balkanski el 2007 , Di Baggio 2017, Ryder et al 2019)
17: Desert dust, bin 1, 0.03-0.55 micron, Composite-Philic (Balkanski et 2007 , Di Baggio 2017, Ryder et al 2019)
18: Desert dust, bin 2, 0.55-0.90 micron, Composite-Philic (Balkanski el 2007 , Di Baggio 2017, Ryder et al 2019)
19: Desert dust, bin 3, 0.90-20.0 micron, Composite-Philic (Balkanski el 2007 , Di Baggio 2017, Ryder et al 2019)
"""
PHOBIC_DESCR = \
"""
1: Desert dust, bin 1, 0.03-0.55 micron, (SW) Dubovik et al. 2002 (LW) Fouquart et al. 1987
2: Desert dust, bin 2, 0.55-0.90 micron, (SW) Dubovik et al. 2002 (LW) Fouquart et al. 1987
3: Desert dust, bin 3, 0.90-20.0 micron, (SW) Dubovik et al. 2002 (LW) Fouquart et al. 1987
4: Desert dust, bin 1, 0.03-0.55 micron, Fouquart et al 1987
5: Desert dust, bin 2, 0.55-0.90 micron, Fouquart et al 1987
6: Desert dust, bin 3, 0.90-20.0 micron, Fouquart et al 1987
7: Desert dust, bin 1, 0.03-0.55 micron, Woodward 2001, Table 2
8: Desert dust, bin 2, 0.55-0.90 micron, Woodward 2001, Table 2
9: Desert dust, bin 3, 0.90-20.0 micron, Woodward 2001, Table 2
10: Hydrophobic organic matter, OPAC (hydrophilic at RH=20%)
11: Black carbon, OPAC
12: Black carbon, Bond and Bergstrom 2006
13: Black carbon, Stier et al 2007
14: Stratospheric sulfate (hydrophilic ammonium sulfate at RH 20%-30%)
15: Desert dust, bin 1, 0.03-0.55 micron, Composite (Balkanski et 2007 , Di Baggio 2017, Ryder et al 2019)
16: Desert dust, bin 2, 0.55-0.90 micron, Composite (Balkanski el 2007 , Di Baggio 2017, Ryder et al 2019)
17: Desert dust, bin 3, 0.90-20.0 micron, Composite (Balkanski el 2007 , Di Baggio 2017, Ryder et al 2019)
18: Hydrophobic organic matter, Brown et al 2018 (hydrophilic at RH=20%)
19: Black carbon, Williams 2007
"""


OPTICS_AERO_MAP = {}

###
## Prognostic 43r3 and Bozzo climatology
###
OPTICS_AERO_MAP[3] = {
    "Sea_Salt_bin1"                   : (1, True),
    "Sea_Salt_bin2"                   : (2, True),
    "Sea_Salt_bin3"                   : (3, True),
    "Mineral_Dust_bin1"               : (7, False), # Composite-Phobic
    "Mineral_Dust_bin2"               : (8, False), # Composite-Phobic
    "Mineral_Dust_bin3"               : (9, False), # Composite-Phobic
    "Organic_Matter_hydrophilic"      : (4, True),
    "Organic_Matter_hydrophobic"      : (4, False),
    "Black_Carbon_hydrophilic"        : (11, False),
    "Black_Carbon_hydrophobic"        : (11, False),
    "Sulfates"                        : (5, True)
}

###
## Prognostic 48r1
###
OPTICS_AERO_MAP[4] = {
    **OPTICS_AERO_MAP[3].copy(),
    **{
    "Nitrate_fine"                    : (9, True),
    "Nitrate_coarse"                  : (10, True),
    "Ammonium"                        : (8, True),
    "Biogenic_Secondary_Organic"      : (6, True),
    "Anthropogenic_Secondary_Organic" : (7, True),
    "Stratospheric_Sulfate"           : (14, False),
    }
}
# Composite phobic dust
OPTICS_AERO_MAP[4]["Mineral Dust_bin1"] = (15, False)
OPTICS_AERO_MAP[4]["Mineral Dust_bin2"] = (16, False)
OPTICS_AERO_MAP[4]["Mineral Dust_bin3"] = (17, False)
# Brown OM
OPTICS_AERO_MAP[4]["Organic_Matter_hydrophilic"] = (11, True)
OPTICS_AERO_MAP[4]["Organic_Matter_hydrophobic"] = (10, False)

###
## IFS-COMPO 48r1-based 4D climatology (Tim's) deployed in IFS 49R2
## has inconsistent sulfates
###
OPTICS_AERO_MAP[41]  = OPTICS_AERO_MAP[4].copy()
OPTICS_AERO_MAP[41]["Sulfates"] = (13, True)

###
## Prognostic 49r1
###
OPTICS_AERO_MAP[5] = OPTICS_AERO_MAP[4].copy()

# Hydrophilic Dust
OPTICS_AERO_MAP[5]["Mineral Dust_bin1"] = (14, True)
OPTICS_AERO_MAP[5]["Mineral Dust_bin2"] = (15, True)
OPTICS_AERO_MAP[5]["Mineral Dust_bin3"] = (16, True)

# Bond BC
OPTICS_AERO_MAP[5]["Mineral Dust_bin3"] = (12, False)

###################
###################

PDIM = "pressure"
HLDPDIM = "half_level_delta_pressure"
HLPDIM = "half_level_pressure"

def get_aero_longname(aerotype, aerohydro, aerobin=None):
    """
        Fetches aerosol long name from properties.
        Returns None if not found.
        If aerobin is not specified and aerotype+aerohydro
        match an aerosol species, the first bin in the bucket is returned
    """
    for long_name,aero in AEROCAMSBUCKET.items():
        if aerotype == aero.spectype and aerohydro == aero.spechydro:
            if aerobin is None or aerobin == aero.specbin:
                return long_name
    return None


def gen_aerosol_dataset(cams_dset, aero_version: int, verbose=False):
    '''
    generate dataset with radiatively-active species and maps to optical properties
    '''

    aero_map = []
    aero_typ = []
    var_to_drop = []

    aeroptics = OPTICS_AERO_MAP[aero_version]

    for aero in cams_dset.data_vars:
        if aero not in aeroptics:
            if aero != PDIM:
                if verbose:
                    print("Warning in gen_aerosol_dataset(): "+\
                          f"Could not find optics version {aero_version} for {aero}."+\
                          "Will be dropped!")
                var_to_drop.append(aero)
            continue
        idx, hydro = aeroptics[aero]
        my_map = -idx if hydro else idx
        aero_map.append(my_map)
        aero_typ.append(aero)

    return cams_dset.drop_vars(var_to_drop)[aero_typ].squeeze(), aero_typ, aero_map

def interpolate_monthly_aerosols(dset: xr.Dataset or xr.DataArray,
                                 dates: xr.DataArray or np.ndarray) -> xr.Dataset or xr.DataArray:
    """
    Interpolation of aerosol monthly climatologies
    """

    prev_month     = (dates.values - np.timedelta64(14,'D')).astype("datetime64[M]")+ np.timedelta64(14,'D')
    foll_month     = (prev_month + np.timedelta64(18,'D')).astype("datetime64[M]") + np.timedelta64(14,'D')

    monthdelta     = foll_month - prev_month
    thisdelta      = dates  - prev_month.astype("datetime64[ns]")

    timeweight     = thisdelta/monthdelta # in [0,1]

    intmonths_bot  = xr.DataArray(data=prev_month.astype("datetime64[ns]"), coords={"time":dates}).dt.month
    intmonths_top  = xr.DataArray(data=foll_month.astype("datetime64[ns]"), coords={"time":dates}).dt.month

    return  (1-timeweight) * dset.sel(month=intmonths_bot).drop_vars("month") +\
        timeweight*dset.sel(month=intmonths_top).drop_vars("month")

def complete_lon_periodic(dset: xr.Dataset or xr.DataArray, method="linear") -> xr.Dataset or xr.DataArray:
    """
    Completes eventually missing 0. and 360. longitude values on periodic domain
    only supported method is linear. If either 0 or 360 are present,
    then those are copied to fill the missing values.
    """
    this_dset = dset.copy()

    minlon = this_dset.lon.min().values
    maxlon = this_dset.lon.max().values

    appendmax = maxlon < 360
    appendmin = minlon > 0


    # Missing 360
    if appendmin or appendmax:
        wei = minlon/(360-maxlon+minlon)
        borderslice = wei*this_dset.sel(lon=maxlon, drop=True) + (1-wei)*this_dset.sel(lon=minlon, drop=True)

    # Missing 0
    if appendmax:
        maxslice = borderslice.assign_coords(lon=360)
        this_dset = xr.concat(
            [this_dset, maxslice], dim="lon")

    if appendmin:
        minslice = borderslice.assign_coords(lon=0)
        this_dset = xr.concat(
            [minslice, this_dset], dim="lon")

    this_dset = this_dset.sortby("lon")

    return this_dset

def complete_lat_boundaries(dset: xr.Dataset or xr.DataArray) -> xr.Dataset or xr.DataArray:
    """
    Completes latitudes 90N and 90S boundary values
    """

    this_dset = dset.copy()

    minlat = this_dset.lat.min().values
    maxlat = this_dset.lat.max().values

    appendmax = maxlat < 90
    appendmin = minlat > -90

    if appendmin:
        minslice = this_dset.sel(lat=minlat).assign_coords(lat=-90)
        this_dset = xr.concat(
            [minslice, this_dset], dim="lat"
        )
    if appendmax:
        maxslice = this_dset.sel(lat=maxlat).assign_coords(lat=90)
        this_dset = xr.concat(
            [this_dset, maxslice], dim="lat"
        )

    this_dset = this_dset.sortby("lat", ascending=False)

    return this_dset


def interpolate_3d_aerosols(aerosol_fields : xr.DataArray, model_pres : xr.DataArray,
global_domain : bool =True):
    import fvertintp_iface as fvint
    import stack_tools as stack


    # Interpolate aerosols horizontal grid
    print("Horizontally interpolating aerosol fields...")
    aerosol_fields = complete_lat_boundaries(
        complete_lon_periodic(aerosol_fields, method="linear")
    )

    model_lons = model_pres["lon"]
    model_lats = model_pres["lat"]
    aerosol_hintp = aerosol_fields.interp(lat=model_lats, lon=model_lons,
                                          method="linear",
                                          kwargs={"fill_value": np.nan}).squeeze().compute()

    # Reorder aerosol fields
    tmp_src = aerosol_hintp[PDIM]
    tmp_dst = model_pres

    tmp_stacktools = stack.tools_to_stack_xarrays(
        src_arr=tmp_src, dst_arr=tmp_dst,
        intp_dim_name="lev")

    tgtlevs, weights = fvint.interp(
        psrc=tmp_src.transpose(*tmp_stacktools.src_dim_order).values.reshape(tmp_stacktools.src_stackshape),
        ptgt=tmp_dst.transpose(*tmp_stacktools.dst_dim_order).values.reshape(tmp_stacktools.dst_stackshape)
    )
    tgtlevs = xr.DataArray(data=tgtlevs.reshape(tmp_stacktools.out_shape),
                            dims=tmp_stacktools.out_dim_order, coords=tmp_stacktools.out_coords)
    weights = xr.DataArray(data=weights.reshape(tmp_stacktools.out_shape),
                            dims=tmp_stacktools.out_dim_order, coords=tmp_stacktools.out_coords)

    del tmp_stacktools, tmp_src, tmp_dst

    aero_mmr = aerosol_hintp["aerosol_mmr"]

    tmp_stacktools = stack.tools_to_stack_xarrays(
        src_arr=aero_mmr, dst_arr=tgtlevs,
        intp_dim_name="lev")

    print("Vertically interpolating aerosol fields...")
    aero_mmr_vintp = xr.DataArray(
        data=fvint.interp_fld(
            fsrc=aero_mmr.transpose(*tmp_stacktools.src_dim_order).values.reshape(tmp_stacktools.src_stackshape),
            tgtlevs=tgtlevs.transpose(*tmp_stacktools.dst_dim_order).values.reshape(tmp_stacktools.dst_stackshape),
            weights=weights.transpose(*tmp_stacktools.dst_dim_order).values.reshape(tmp_stacktools.dst_stackshape)
            ).reshape(tmp_stacktools.out_shape),
        dims=tmp_stacktools.out_dim_order, coords=tmp_stacktools.out_coords
    )

    print("Interpolation done")
    return aero_mmr_vintp
