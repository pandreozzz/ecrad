import numpy as np
import xarray as xr

# Magic constants
RNAVO = 6.0221367E+23
RKBOL = 1.380658E-23
RMD   = 28.9644
R     = RNAVO*RKBOL
RD    = 1000*R/RMD

def compute_ml_pressure(dset : xr.Dataset, half_level=False):
    """
    lnsp required
    """
    p_sfc = np.exp(dset["lnsp"])

    aname = "hyai" if half_level else "hyam"
    bname = "hybi" if half_level else "hybm"
    levstart = 0 if half_level else 1
    levname = "half_level" if half_level else "lev"

    if (aname in dset) and (bname in dset):
        coef_a, coef_b = dset[aname], dset[bname]
        dimname = coef_a.dims[0]
        nlevs   = len(coef_a[dimname])
    else:
        raise ValueError("Could not find coefficients to compute pressure on model levels.")

    levend = nlevs if half_level else nlevs+1
    levcoord = xr.DataArray(
        data=np.arange(levstart,levend, dtype=np.float64),
        dims=[levname,],
        attrs={
            "long_name" : "Layer number",
            "positive"  : "down"})

    p_lev = (coef_a + coef_b*p_sfc).rename({dimname : levname}).assign_coords({levname: levcoord})

    return p_lev

def compute_hl_temperature(p_half : xr.DataArray, p_full : xr.DataArray,
                           t_full : xr.DataArray, skt_sfc : xr.DataArray):
    """
    p_half : pressure at half level
    """

    if "lev" in t_full.dims:
        lev_dim = "lev"
    elif "level" in t_full.dims:
        lev_dim = "level"
    else:
        raise ValueError("Error in compute_hl_temperature: could not find any level dimension in t_full!")
    if lev_dim not in p_full.dims:
        raise ValueError(f"Error in compute_hl_temperature: could not find {lev_dim} dimension in p_full!")

    # number of levels
    n_lev = len(t_full[lev_dim])

    # Create destination field
    t_half = xr.full_like(p_half, 0.)

    # Ensure same dimension order to arrays in calculations
    dimlist  = list(p_full.squeeze(drop=True).dims)
    dimlist_flat = dimlist.copy()
    if lev_dim in dimlist_flat:
        dimlist_flat.remove(lev_dim)


    # Extract temperature at full levels and skin temperature (sfc field)
    t_full   = t_full.transpose(*dimlist)
    skt_sfc  = skt_sfc.transpose(*dimlist_flat)

    # These levels numbers corespond to sandwiched half levels e.g (1,2,...,136)
    inner_levs = np.arange(1,n_lev)

    p_half_renamed = p_half.rename(half_level=lev_dim)
    t_half_renamed = t_half.rename(half_level=lev_dim)
    # Generate values required by eq. 2.52 of Documentation CY48r1 part IV
    t_fl       = t_full.sel({lev_dim:inner_levs})    # k
    t_fl_plus1 = t_full.sel({lev_dim:inner_levs+1})  # k+1
    t_fl_plus1[lev_dim] = inner_levs
    p_fl       = p_full.sel({lev_dim:inner_levs})    # k
    p_fl_plus1 = p_full.sel({lev_dim:inner_levs+1})  # k+1
    p_fl_plus1[lev_dim] = inner_levs
    p_hl       = p_half_renamed.sel({lev_dim:inner_levs})    # k+1/2

    # Fill inner levels e.g (exclude 0 and 137)
    t_half_renamed[{lev_dim : inner_levs}] = (t_fl*p_fl*(p_fl_plus1 - p_hl) + t_fl_plus1*p_fl_plus1*(p_hl-p_fl))/(p_hl*(p_fl_plus1-p_fl))
    logp_nth_hl   = np.log(p_half_renamed.sel({lev_dim:n_lev}))
    logp_nmin1_hl = np.log(p_half_renamed.sel({lev_dim:n_lev-1}))
    logp_nmin1_hl[lev_dim] = n_lev
    logp_nth_fl   = np.log(p_full.sel({lev_dim:n_lev}).values)

    t_nth_fl      = t_full.sel({lev_dim:n_lev})
    t_nmin1_hl    = t_half_renamed.sel({lev_dim:n_lev-1})
    t_nmin1_hl[lev_dim] = n_lev

    # Fill top layer e.g. 0
    t_half_renamed[{lev_dim : 0}] = t_full.sel({lev_dim:1}).values

    # Fill closest-to-surface layer (n+1/2) e.g. 137
    t_half_renamed[{lev_dim : n_lev}] = 0.5 * (skt_sfc.values + t_nth_fl+ (t_nth_fl - t_nmin1_hl)/(logp_nth_fl - logp_nmin1_hl)*(logp_nth_hl - logp_nth_fl))

    return t_half_renamed.rename("temperature_hl").rename({lev_dim : "half_level"})

def compute_ice_reff_ifs(dset, latitude = None, default_re :float = 10., min_ice : int = 1,
                         max_diameter : float = 155, rre2de : float = 0.64952):
    """
    Reproduced computations of the ice effective radius as in IFS (RADIP = 3)
    dset xarray.Dataset : must contain the following fields:
                          * cc         -> grid-point cloud cover
                          * ciwc, cswc -> cloud ice/snow water content
                          * lsm        -> land fraction
                          * p          -> full-level pressure
                          * t          -> full-level temperature

    Returns:
    --------
    xarray.DataArray "re_ece" containing ice effective radius in micrometers.
    """

    if latitude is None:
        if "latitude" in dset.coords:
            lats = dset.latitude
        elif "lat" in dset.coords:
            lats = dset.lat
        else:
            raise ValueError(f"Error in compute_ice_reff_ifs: could not finde latitude coords in dset!")

    rtt = 273.15

    if min_ice != 0:
        min_diameter = 20 + (float(min_ice) - 20)*np.cos(lats)
    else:
        min_diameter = 0.

    # Do computations where clouds are
    mask        = np.logical_and(dset["cc"] >= 0.001, (dset["ciwc"]+dset["cswc"]) > 0)
    cloudy_dset = dset.where(mask)

    # Compute liquid and rain water contents
    air_density_gm3 = (1000*cloudy_dset["p"]/(cloudy_dset["t"]*RD)).rename("density")
    iwc_incloud_gm3 = air_density_gm3*(cloudy_dset["ciwc"]+cloudy_dset["cswc"])/cloudy_dset["cc"]

    temp_celsius    = cloudy_dset["t"] - rtt

    aiwc = 25.8966 * iwc_incloud_gm3**0.2214
    biwc = 0.7957  * iwc_incloud_gm3**0.2535


    diameter_um = (1.2351 + 0.0105 * temp_celsius) * (aiwc + biwc*(dset["t"] -83.15))

    diameter_um = diameter_um.clip(min_diameter, max_diameter)
    reff = diameter_um * rre2de

    return xr.where(mask, reff, default_re)

def compute_ccn_ifs(ws :xr.DataArray, lsm : xr.DataArray):
    """
    Ws : absolute 10m wind speed
    lsm : land-sea mask
    """
    landmask = lsm > 0.5
    seamask  = np.logical_not(landmask)
    wind_lt15 = ws < 15
    wind_lt30 = ws < 30

    a_par = xr.where(wind_lt15, 0.16, 0.13)
    b_par = xr.where(wind_lt15, 1.45, 1.89)
    qa = np.clip(np.exp(a_par*ws+b_par), -np.inf, 327)

    c_par = xr.where(landmask, 2.21, 1.2)
    d_par = xr.where(landmask, 0.3,  0.5)
    na = 10**(c_par + d_par*np.log10(qa))

    nd = xr.where(landmask,
                  -2.10e-4*na**2 + 0.568*na - 27.9,
                  -1.15e-3*na**2 + 0.963*na + 5.30
                 )

    return nd

def compute_liquid_reff_ifs(dset, ccn_fields = None, wood_correction : bool = True,
                        min_reff : float = 4., max_reff : float = 30., cle_reff : float = 2.,
                        min_ccn: float = 1., max_ccn: float = 3000.,
                        spectr_disp_land : float = 0.69, spectr_disp_sea : float = 0.77):
    """
    Reproduced computation of the liquid effective radius as in IFS (RADLP = 2)
    dset xarray.Dataset : must contain the following fields:
                          * cc         -> grid-point cloud cover
                          * clwc, crwc -> cloud liquid/rain water content
                          * lsm        -> land fraction
                          * p          -> full-level pressure
                          * t          -> full-level temperature

    Returns:
    --------
    xarray.DataArray "re_liquid" containing liquid effective radius in micrometers.
    """

    # Global IFS variables STILL TO FILL
    repscw = 1.e-12 #sec. epsilon for abs. amount in laplace transform
    replog = 1.e-12 #sec. epsilon for cloud liquid water path

    mask     = np.logical_and(dset["cc"] >= 0.001, (dset["clwc"]+dset["crwc"]) > 0)

    cloudy_dset = dset.where(mask)

    # Spectral dispersion (land vs. sea) (ZSPECTRAL_DISPERSION)
    spectr_disp = xr.where(dset["lsm"] > 0.5, spectr_disp_land, spectr_disp_sea).where(mask)
    ratio       = (0.222/spectr_disp)**0.333

    # Compute liquid and rain water contents
    air_density_gm3 = 1000 * (cloudy_dset["p"]/(cloudy_dset["t"]*RD)).rename("density")
    # In-cloud mean water contents found by dividing by cloud
    # fraction
    lwc_gm3     = air_density_gm3 * cloudy_dset["clwc"] / cloudy_dset["cc"]
    rwc_gm3     = air_density_gm3 * cloudy_dset["crwc"] / cloudy_dset["cc"]

    # Where to get cdcn_fields from
    if ccn_fields is None:
        pot_cdnc = 150
    else:
        pot_cdnc = np.clip(ccn_fields.where(mask), min_ccn, max_ccn)

    if wood_correction:
        # Wood's (2000, eq. 19) adjustment to Martin et al's
        # parametrization (ZWOOD_FACTOR)
        wood_mask   = lwc_gm3 > repscw
        rain_ratio  = xr.where(wood_mask, rwc_gm3/lwc_gm3.where(wood_mask), 0.)
        wood_factor = xr.where(wood_mask,
                               (1 + rain_ratio)**(0.666)/(1+0.2*ratio*rain_ratio),
                               1).rename("wood_factor")
    else:
        wood_factor = 1.

    # g m-3 and cm-3 units cancel out with density of water
    # 10^6/(1000*1000); need a factor of 10^6 to convert to
    # microns and cubed root is factor of 100 which appears in
    # equation below
    re_cubed = (3 * (lwc_gm3 + rwc_gm3))/(4*np.pi*pot_cdnc*spectr_disp)

    replog_mask = re_cubed > replog
    reff        = xr.where(replog_mask,
                           wood_factor*100*re_cubed**0.333,
                           min_reff).clip(min_reff, max_reff).rename("re_liquid")

    # fill non-cloud areas with clear values and return fields in micrometers
    return xr.where(mask, reff, cle_reff)
