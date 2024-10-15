from collections import namedtuple
import numpy as np
import xarray as xr

def compute_lut_species_from_ifs_species(recipes: dict, aero_cams: xr.Dataset) -> xr.Dataset:
    """
        recipes and aero_cams and get the lut species
    """
    lut_species = {}
    for r,recipe in enumerate(recipes):
        this_ingredients = []
        for aero in recipe.keys():
            if aero in aero_cams.aer_type.values:
                this_ingredients.append(
                    (recipe[aero]*aero_cams.sel(aer_type=aero, drop=True)).expand_dims(dim="tmp_ingredient"))
            else:
                print(f"Warning: {aero} not found in aerosol fields!")

        this_bowl = xr.concat(this_ingredients, coords="all",
                              dim="tmp_ingredient").sum(dim="tmp_ingredient")

        lut_species[f"aero{r+1}"] = (this_bowl).rename(f"aero{r+1}_mcon")

        del this_bowl

    return xr.Dataset(data_vars=lut_species)


#############################
# READ PYRCEL LOOKUP TABLE
#############################

def get_dics_from_recipes_dset(rec_ds: xr.Dataset) -> dict:
    """
    Unpacks recipes dataset into a dictionary
    """
    from aeromaps import get_aero_longname

    specieslist = [var for var in rec_ds.variables if "aero" in var]
    aeronames = xr.full_like(rec_ds.ifs_species, "", dtype="object")
    aeronames[:] = [get_aero_longname(ty,hy,bi)
                 for ty,hy,bi in zip(rec_ds.ifs_types, rec_ds.ifs_hydro, rec_ds.ifs_bins)]

    recipe_dics = []
    for spec in specieslist:
        # Consider only contributing aerosols to this lut species
        thisrec = rec_ds[spec].where(rec_ds[spec] > 0).dropna("ifs_species")
        thisspc = thisrec.ifs_species.values
        recipe_dics.append({aeronames.sel(ifs_species=sp).values.item(): thisrec.sel(ifs_species=sp).values.item()
                            for sp in thisspc})
    return recipe_dics

NumpyLut = namedtuple(
    "Numpy_LUT",
    ["aero_order", "lut_map", "lut_map_sep",
     "lut_mass_bins", "lut_ncon_bins"])
def get_numpy_lut(lut_dset):
    """
        Returns ordered lut elements (aero 1 to 5)
    """

    aero_order = [f"aero{i}" for i in range(1,5)]
    mass_order = [f"{ae}_mass" for ae in aero_order]
    ncon_order = [f"{ae}_nccn" for ae in aero_order]

    lut_map = lut_dset["total_cdnc"].transpose(*mass_order).values

    lut_map_sep = np.concatenate(
        [lut_dset[f"aero{i}_cdnc"].transpose(*mass_order).values[None,:]
         for i in range(1,5)],
        axis=0)

    lut_mass_bins = np.concatenate(
        [lut_dset[coord].values[None,:] for coord in mass_order],
        axis=0)
    lut_ncon_bins = np.concatenate(
        [lut_dset[coord].values[None,:] for coord in ncon_order],
        axis=0)

    return NumpyLut(aero_order, lut_map, lut_map_sep,
                    lut_mass_bins, lut_ncon_bins)


def compute_cdnc(lut_species, lut_dset,#lut_version = "v3",
                 min_cdnc = 10, max_cdnc = 1800,
                 with_extra_fields: bool = False):

    import fget_lutval_iface

    #lut_dset = xr.open_dataset(LUTPATH[lut_version]).load()
    numpy_lut = get_numpy_lut(lut_dset)

    # Last dimension is species
    #print("Reordering lut_species...")
    np_lut_species = np.concatenate(
        [lut_species[aero].values[...,None] for aero in numpy_lut.aero_order],
        axis=-1)


    #print("Calling get_cdnc...")
    ndroplets = fget_lutval_iface.get_cdnc(
        np_lut_species, numpy_lut.lut_map, numpy_lut.lut_mass_bins,
        numpy_lut.lut_map_sep, numpy_lut.lut_ncon_bins,
        with_extra_fields=with_extra_fields)

    #print("Postprocessing...")
    # Xarray
    species_mould = xr.zeros_like(lut_species[numpy_lut.aero_order[0]])

    cdnc_total = species_mould.copy()
    cdnc_total[...] = ndroplets.total_cdnc
    data_vars = {"cdnc" : cdnc_total}

    if with_extra_fields:
        for s,spec in enumerate(numpy_lut.aero_order):
            this_binnum = species_mould.copy()
            this_binnum[...] = ndroplets.bin_number[...,s]
            this_ccnnum = species_mould.copy()
            this_ccnnum[...] = ndroplets.ccn_number[...,s]
            this_ccnact = species_mould.copy()
            this_ccnact[...] = ndroplets.activated_fraction[...,s]

            data_vars[f"{spec}_bin"] = this_binnum
            data_vars[f"{spec}_ccn"] = this_ccnnum
            data_vars[f"{spec}_act"] = this_ccnact

            del this_binnum, this_ccnnum, this_ccnact


    return xr.Dataset(data_vars = data_vars)
