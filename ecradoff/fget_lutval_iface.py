from collections import namedtuple
import ctypes as ct
import numpy as np

f_lib = ct.CDLL('./f_src/fget_lutval.so')
c_real = ct.c_double
c_real_ptr = np.ctypeslib.ndpointer(c_real)#ct.POINTER(c_real)
c_int  = ct.c_int32
c_int_ptr = np.ctypeslib.ndpointer(c_int)#ct.POINTER(c_int)

f_lib.get_cdnc.argtypes = [c_real_ptr,]*4+[c_int,]*5
f_lib.get_cdnc_sep.argtypes = [c_real_ptr,]*4+[c_int,]*5 + [c_real_ptr,]*2 + [c_int_ptr,] + [c_real_ptr,]*2
f_lib.get_bin_nearest.argtypes = [c_real_ptr,]*2+[c_int_ptr,]+[c_int,]*2

def get_bin_nearest(species_mass, mass_bins):
    """
        species_mass(nspec)
        mass_bins(nspec,nbins)
    """
    nspec = len(species_mass)
    nspec2,nbins = mass_bins.shape

    if (nspec != nspec2):
        raise ValueError("Inconsistent number of species in values and bins." +\
                         f"species_mass ({nspec}) and mass_bins({nbins},{nspec2})")


    species_mass = np.ascontiguousarray(species_mass)
    mass_bins = np.ascontiguousarray(mass_bins)
    tgtbins = np.ascontiguousarray(np.zeros_like(species_mass, dtype=c_int))

    f_lib.get_bin_nearest(
        species_mass.astype(c_real),
        mass_bins.astype(c_real),
        tgtbins,
        c_int(nspec), c_int(nbins))

    return tgtbins

NDroplets = namedtuple(
    "LUT_CDNC",
    ["total_cdnc", "bin_number",
     "ccn_number", "activated_fraction"])

def get_cdnc(species_mass, lut_map, lut_mass_bins,
             lut_map_sep=None, lut_ncon_bins=None, with_extra_fields=False):
    """
        Interpolates psrc to ptgt
        species_mass(nx,ny,nz,nspec)
        lut_map(nbins,nbins,nbins,nbins)
        lut_mass_bins(nspec,nbins)

        returns
        cdnc(nx,ny,nz,nlevtgt)
    """
    nx,ny,nz,nspec  = species_mass.shape
    nb1,nb2,nb3,nb4 = lut_map.shape
    nspec2, nbins = lut_mass_bins.shape

    if (nspec != 4) or (nspec2 != 4):
        raise ValueError("Only 4 species supported!!" +
                        f"species_mass({nspec}), lut_mass_bins({nspec2})")


    if (nb1 != nb2) or (nb3 != nb4) or (nb1 != nb3) or (nb1 != nbins):
        raise ValueError(
                "Number of bins must always be the same!"+\
                f"lat_map({nb1},{nb2},{nb3},{nb4}) lut_mass_bins({nbins})")

    cdnc_out = np.zeros_like(species_mass[...,0], dtype=c_real)

    # Enforce contiguity
    species_mass  = np.ascontiguousarray(species_mass)
    lut_map       = np.ascontiguousarray(lut_map)
    lut_mass_bins = np.ascontiguousarray(lut_mass_bins)
    cdnc_out      = np.ascontiguousarray(cdnc_out)

    if with_extra_fields:
        nspec2,nb1,nb2,nb3,nb4 = lut_map_sep.shape
        if (nb1 != nb2) or (nb3 != nb4) or (nb1 != nb3) or (nb1 != nbins):
            raise ValueError(
                    "Number of bins must always be the same!"+\
                    f"lat_map_sep({nb1},{nb2},{nb3},{nb4},{nspec}) lut_mass_bins({nbins})")
        if (nspec != nspec2):
            raise ValueError("Number of species must be the same!"
                             f"{nspec} in species_mass, {nspec2} in lut_map_sep")


        lut_map_sep   = np.ascontiguousarray(lut_map_sep)
        lut_ncon_bins = np.ascontiguousarray(lut_ncon_bins)
        bin_num =  np.ascontiguousarray(np.zeros_like(species_mass, dtype=c_int))
        ccn_num = np.ascontiguousarray(np.zeros_like(species_mass, dtype=c_real))
        ccn_act = np.ascontiguousarray(np.zeros_like(species_mass, dtype=c_real))

        f_lib.get_cdnc_sep(
            species_mass.astype(c_real),
            lut_map.astype(c_real),
            lut_mass_bins.astype(c_real),
            cdnc_out,
            c_int(nx), c_int(ny), c_int(nz),
            c_int(nspec), c_int(nbins),
            lut_map_sep.astype(c_real),
            lut_ncon_bins.astype(c_real),
            bin_num,
            ccn_num,
            ccn_act
            )
        return NDroplets(cdnc_out, bin_num, ccn_num, ccn_act)
    else:
        f_lib.get_cdnc(
                species_mass.astype(c_real),
                lut_map.astype(c_real),
                lut_mass_bins.astype(c_real),
                cdnc_out,
                c_int(nx), c_int(ny), c_int(nz),
                c_int(nspec), c_int(nbins)
                )
        return NDroplets(cdnc_out, None, None, None)

if __name__=="__main__":
    print("nothing to do here")
