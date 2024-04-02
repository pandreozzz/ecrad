import xarray as xr
from dataclasses import dataclass

#############################
# CAMS SPECIES NAMES
#############################
@dataclass
class ClimSpecies:
    longname  : str
    shortname : str
    spectype  : str
    spechydro : bool
    specbin   : int
    def __post_init__(self):
        for (name, field_type) in self.__annotations__.items():
            if not isinstance(self.__dict__[name], field_type):
                current_type = type(self.__dict__[name])
                raise TypeError(f"The field `{name}` was assigned by `{current_type}` instead of `{field_type}`")

AEROCAMSBUCKET = {
    "Sea_Salt_bin1"                   : ClimSpecies("Sea_Salt_bin1",              
                                                    "SS1", "SS",  True, 1),
    "Sea_Salt_bin2"                   : ClimSpecies("Sea_Salt_bin2",              
                                                    "SS2", "SS",  True, 2),
    "Sea_Salt_bin3"                   : ClimSpecies("Sea_Salt_bin3",              
                                                    "SS3", "SS",  True, 3),
    "Mineral_Dust_bin1"               : ClimSpecies("Mineral_Dust_bin1",          
                                                    "DD1", "DD", False, 1),
    "Mineral_Dust_bin2"               : ClimSpecies("Mineral_Dust_bin2",          
                                                    "DD2", "DD", False, 2),
    "Mineral_Dust_bin3"               : ClimSpecies("Mineral_Dust_bin3",          
                                                    "DD3", "DD", False, 3),
    "Organic_Matter_hydrophilic"      : ClimSpecies("Organic_Matter_hydrophilic", 
                                                    "OMH", "OM",  True, 0),
    "Organic_Matter_hydrophobic"      : ClimSpecies("Organic_Matter_hydrophobic", 
                                                    "OMN", "OM", False, 0),
    "Black_Carbon_hydrophilic"        : ClimSpecies("Black_Carbon_hydrophilic",   
                                                    "BCH", "BC",  True, 0),
    "Black_Carbon_hydrophobic"        : ClimSpecies("Black_Carbon_hydrophobic",   
                                                    "BCN", "BC", False, 0),
    "Sulfates"                        : ClimSpecies("Sulfates",                   
                                                    "SU", "SU",  True, 0),
    "Nitrate_fine"                    : ClimSpecies("Nitrate_fine",               
                                                    "NI1", "NI",  True, 1),
    "Nitrate_coarse"                  : ClimSpecies("Nitrate_coarse",             
                                                    "NI2", "NI",  True, 2),
    "Ammonium"                        : ClimSpecies("Ammonium",                   
                                                    "AM", "AM",  True, 0),  
    "Biogenic_Secondary_Organic"      : ClimSpecies("Biogenic_Secondary_Organic", 
                                                    "BSO", "OB",  True, 0),  
    "Anthropogenic_Secondary_Organic" : ClimSpecies("Anthropogenic_Secondary_Organic",
                                                    "ASO", "OA",  True, 0),
    "Stratospheric_Sulfate"           : ClimSpecies("Stratospheric_Sulfate",
                                                    "SSU", "SSU",  False, 0),  
}

AEROPTICSFILE = "aerosol_ifs_49R1_20230725.nc"
AEROPTICSMAP = {}
AEROMAPDESCR = {}

PDIM = "pressure"
HLDPDIM = "half_level_delta_pressure"
HLPDIM = "half_level_pressure"

AEROMAPDESCR[3] = """ (CY48R1 with 43r3 climatology)
    Aerosol mapping:
       1 -> hydrophilic type 1: Sea salt, bin 1, 0.03-0.5 micron, OPAC
       2 -> hydrophilic type 2: Sea salt, bin 2, 0.50-5.0 micron, OPAC
       3 -> hydrophilic type 3: Sea salt, bin 3, 5.0-20.0 micron, OPAC
       4 -> hydrophobic type 7: Desert dust, bin 1, 0.03-0.55 micron, Woodward 2001, Table 2
       5 -> hydrophobic type 8: Desert dust, bin 2, 0.55-0.90 micron, Woodward 2001, Table 2
       6 -> hydrophobic type 9: Desert dust, bin 3, 0.90-20.0 micron, Woodward 2001, Table 2
       7 -> hydrophilic type 4: Hydrophilic organic matter, OPAC
       8 -> hydrophobic type 10: Hydrophobic organic matter, OPAC (hydrophilic at RH=20%)
       9 -> hydrophobic type 11: Black carbon, OPAC
      10 -> hydrophobic type 11: Black carbon, OPAC
      11 -> hydrophilic type 5: Ammonium sulfate (for sulfate), GACP Lacis et al https://gacp.giss.nasa.gov/data_sets/
      12 -> hydrophobic type 10: Hydrophobic organic matter, OPAC (hydrophilic at RH=20%) (TROPOSPHERIC BACKGROND ORGANIC)
      13 -> hydrophobic type 14: Stratospheric sulfate, GACP (hydrophilic ammonium sulfate at RH 20%-30%)
"""
AEROPTICSMAP[3] = {
    "Sea_Salt_bin1"                   : (1, True),
    "Sea_Salt_bin2"                   : (2, True),
    "Sea_Salt_bin3"                   : (3, True),
    "Mineral_Dust_bin1"               : (7, False), # Dust Woodward!
    "Mineral_Dust_bin2"               : (8, False),
    "Mineral_Dust_bin3"               : (9, False),
    "Organic_Matter_hydrophilic"      : (4, True), # OPAC
    "Organic_Matter_hydrophobic"      : (10, False), # OPAC
    "Black_Carbon_hydrophilic"        : (11, False),
    "Black_Carbon_hydrophobic"        : (11, False),
    "Sulfates"                        : (5, True), # GACP
    "Stratospheric_Sulfate"           : (14, False),  
}

AEROMAPDESCR[4] = """ (CY48R1 with 49r2 climatology)
    Aerosol mapping:
       1 -> hydrophilic type 1: Sea salt, bin 1, 0.03-0.5 micron, OPAC
       2 -> hydrophilic type 2: Sea salt, bin 2, 0.50-5.0 micron, OPAC
       3 -> hydrophilic type 3: Sea salt, bin 3, 5.0-20.0 micron, OPAC
       4 -> hydrophobic type 15: Desert dust, bin 1, 0.03-0.55 micron, Composite (Balkanski et 2007 , Di Baigo 2017, Ryder et al 2019)
       5 -> hydrophobic type 16: Desert dust, bin 2, 0.55-0.90 micron, Composite (Balkanski el 2007 , Di Baigo 2017, Ryder et al 2019)
       6 -> hydrophobic type 17: Desert dust, bin 3, 0.90-20.0 micron, Composite (Balkanski el 2007 , Di Baigo 2017, Ryder et al 2019)
       7 -> hydrophilic type 11: Hydrophilic organic matter, Brown et al 2018
       8 -> hydrophobic type 18: Hydrophobic organic matter, Brown et al 2018 (hydrophilic at RH=20%)
       9 -> hydrophobic type 11: Black carbon, OPAC
      10 -> hydrophobic type 11: Black carbon, OPAC
      11 -> hydrophilic type 13: Sulfate, GACP Lacis et al https://gacp.giss.nasa.gov/data_sets/ with modified size distribution
      12 -> hydrophilic type 9: Fine mode Nitrate, GLOMAP
      13 -> hydrophilic type 10: Coarse mode Nitrate, GLOMAP
      14 -> hydrophilic type 8: Fine mode Ammonium sulfate (for ammonia), GACP Lacis et al https://gacp.giss.nasa.gov/data_sets/
      15 -> hydrophilic type 6: Secondary organic aerosol - biogenic, Moise et al 2015
      16 -> hydrophilic type 7: Secondary organic aerosol - anthropogenic, Moise et al 2015
      17 -> hydrophobic type 18: Hydrophobic organic matter, Brown et al 2018 (hydrophilic at RH=20%)  (TROPOSPHERIC BACKGROND ORGANIC)
      18 -> hydrophobic type 14: Stratospheric sulfate, GACP (hydrophilic ammonium sulfate at RH 20%-30%)
"""
AEROPTICSMAP[4] = {
    "Sea_Salt_bin1"                   : (1, True),
    "Sea_Salt_bin2"                   : (2, True),
    "Sea_Salt_bin3"                   : (3, True),
    "Mineral_Dust_bin1"               : (15, False), # Composite-Phobic
    "Mineral_Dust_bin2"               : (16, False), # Composite-Phobic
    "Mineral_Dust_bin3"               : (17, False), # Composite-Phobic
    "Organic_Matter_hydrophilic"      : (11, True),
    "Organic_Matter_hydrophobic"      : (18, False),
    "Black_Carbon_hydrophilic"        : (11, False),
    "Black_Carbon_hydrophobic"        : (11, False),
    "Sulfates"                        : (13, True),
    "Nitrate_fine"                    : (9, True),
    "Nitrate_coarse"                  : (10, True),
    "Ammonium"                        : (8, True),  
    "Biogenic_Secondary_Organic"      : (6, True),  
    "Anthropogenic_Secondary_Organic" : (7, True),
    "Stratospheric_Sulfate"           : (14, False),  
}

AEROMAPDESCR[5] = """
    Aerosol mapping:
       1 -> hydrophilic type 1: Sea salt, bin 1, 0.03-0.5 micron, OPAC
       2 -> hydrophilic type 2: Sea salt, bin 2, 0.50-5.0 micron, OPAC
       3 -> hydrophilic type 3: Sea salt, bin 3, 5.0-20.0 micron, OPAC
       4 -> hydrophilic type 14: Desert dust, bin 1, 0.03-0.55 micron, Composite-Philic Non-Sphere-Scaling-Kandler (Balkanski et 2007 , Di Baggio 2017, Ryder et al 2019)
       5 -> hydrophilic type 15: Desert dust, bin 2, 0.55-0.90 micron, Composite-Philic Non-Sphere-Scaling-Kandler (Balkanski el 2007 , Di Baggio 2017, Ryder et al 2019)
       6 -> hydrophilic type 16: Desert dust, bin 3, 0.90-20.0 micron, Composite-Philic Non-Sphere-Scaling-Kandler (Balkanski el 2007 , Di Baggio 2017, Ryder et al 2019)
       7 -> hydrophilic type 11: Hydrophilic organic matter, Brown et al 2018
       8 -> hydrophobic type 18: Hydrophobic organic matter, Brown et al 2018 (hydrophilic at RH=20%)
       9 -> hydrophobic type 12: Black carbon, Bond and Bergstrom 2006
      10 -> hydrophobic type 12: Black carbon, Bond and Bergstrom 2006
      11 -> hydrophilic type 13: Sulfate, GACP Lacis et al https://gacp.giss.nasa.gov/data_sets/ with modified size distribution
      12 -> hydrophilic type 9: Fine mode Nitrate, GLOMAP
      13 -> hydrophilic type 10: Coarse mode Nitrate, GLOMAP
      14 -> hydrophilic type 8: Fine mode Ammonium sulfate (for ammonia), GACP Lacis et al https://gacp.giss.nasa.gov/data_sets/
      15 -> hydrophilic type 6: Secondary organic aerosol - biogenic, Moise et al 2015
      16 -> hydrophilic type 7: Secondary organic aerosol - anthropogenic, Moise et al 2015
      17 -> hydrophobic type 18: Hydrophobic organic matter, Brown et al 2018 (hydrophilic at RH=20%)  (TROPOSPHERIC BACKGROND ORGANIC)
      18 -> hydrophobic type 14: Stratospheric sulfate (hydrophilic ammonium sulfate at RH 20%-30%)
"""
AEROPTICSMAP[5] = {
    "Sea_Salt_bin1"                   : (1,  True),
    "Sea_Salt_bin2"                   : (2,  True),
    "Sea_Salt_bin3"                   : (3,  True),
    "Mineral_Dust_bin1"               : (14, True), # Composite-Philic Non-Sphere-Scaling-Kandler
    "Mineral_Dust_bin2"               : (15, True),
    "Mineral_Dust_bin3"               : (16, True),
    "Organic_Matter_hydrophilic"      : (11, True), # Brown 2018
    "Organic_Matter_hydrophobic"      : (18, False), # Brown 2018
    "Black_Carbon_hydrophilic"        : (12, False),# Bond 2006
    "Black_Carbon_hydrophobic"        : (12, False),
    "Sulfates"                        : (13, True), # GACP-NewPSD
    "Nitrate_fine"                    : (9,  True),
    "Nitrate_coarse"                  : (10, True),
    "Ammonium"                        : (8,  True),  
    "Biogenic_Secondary_Organic"      : (6,  True),  
    "Anthropogenic_Secondary_Organic" : (7,  True),
    "Stratospheric_Sulfate"           : (14, False),  
}

def gen_aerosol_dataset(cams_dset, aero_version : int, verbose=False):
    '''
    generate dataset with radiatively-active species and maps to optical properties 
    '''

    aero_map = []
    aero_typ = []
    var_to_drop = []

    aeroptics = AEROPTICSMAP[aero_version]

    for aero in cams_dset.data_vars:
        if aero not in aeroptics:
            if aero != PDIM:
                if verbose:
                    print("Warning in gen_aerosol_dataset(): "+\
                          f"Could not find optics version {aero_version} for {aero}."+\
                          "Will be dropped!")
                var_to_drop.append(aero)
            continue
        else:
            idx,hydro = aeroptics[aero]
            my_map = -idx if hydro else idx
            aero_map.append(my_map)
            aero_typ.append(aero)

    return cams_dset.drop_vars(var_to_drop)[aero_typ].squeeze(), aero_typ, aero_map

def interpolate_monthly_aerosols(dset : xr.Dataset or xr.DataArray,
                             dates : xr.DataArray or np.ndarray) -> xr.Dataset or xr.DataArray:
    import numpy  as np

    da_dttype = "datetime64[D]"
    mo_dttype = "datetime64[M]"
    ns_dttype = "datetime64[ns]"

    one_day = np.timedelta64(1,'D')
        
    prev_month     = (dates - 14*one_day).astype(mo_dttype) + 14*one_day
    foll_month     = (prev_month + 18*one_day).astype(mo_dttype) + 14*one_day

    monthdelta     = foll_month - prev_month
    thisdelta      = dates  - prev_month

    timeweight     = thisdelta/monthdelta # in [0,1]

    intmonths_bot  = prev_month.dt.month
    intmonths_top  = foll_month.dt.month

    intpdset =  (1-timeweight) * dset.sel(month=intmonths_bot).drop_vars("month") +\
        timeweight*dset.sel(month=intmonths_top).drop_vars("month")
    
    return intpdset.assign_coords(time=dates)

def interpolate_3d_aerosols(aerosol_fields, model_pres):
    import fvertintp_iface

    model_lons = model_pres["lon"]
    model_lats = model_pres["lat"]

    # Interpolate aerosols horizontal grid
    print("Horizontally interpolating aerosol fields...")
    aerosol_hintp = aerosol_fields.interp(lat=model_lats, lon=model_lons,
                                           method="linear",
                                          kwargs={"fill_value": "extrapolate"}).squeeze().compute()
    # Reorder aerosol fields
    aero_dim = list(set(aerosol_hintp.dims) - set(model_pres.dims))[0]
    aerosol_hintp = aerosol_hintp.transpose(aero_dim, "lon", "lat", "lev")
    naero_flds = len(aerosol_hintp[aero_dim]) 
    aero_mmr  = aerosol_hintp["aerosol_mmr"]
    aero_pres = aerosol_hintp[PDIM]

    # Reorder model fields
    model_pres = model_pres.transpose("lon", "lat", "lev")

    # Get level maps and weights
    print("Computing weights for vertical interpolation...")
    l, w = fvertintp_iface.interp(aero_pres, model_pres)

    # Interpolate aerosol fields
    print("Vertically interpolating aerosol fields...")
    aerosol_mmr_vintp = xr.concat(
        [xr.zeros_like(model_pres).expand_dims(aero_dim, axis=0),]*naero_flds,
        dim=aero_dim)
    aerosol_mmr_vintp[...] = fvertintp_iface.interp_fld(aero_mmr[...], tgtlevs=l, weights=w)

    # When fetching points out of the original grid domain,
    # lienar extrapolation can result in nonphysical values.
    # this is a temporary fix and should be removed by extending original domain
    # to simulate proper wrapping and remove extrapolation 
    aerosol_mmr_vintp = aerosol_mmr_vintp.clip(0., 1.)


    return aerosol_mmr_vintp
