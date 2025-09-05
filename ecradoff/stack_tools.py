import numpy as np

from collections import namedtuple
StackTools = namedtuple('StackTools', ["src_dim_order", "dst_dim_order", "out_dim_order",
                                       "src_stackshape", "dst_stackshape",
                                       "out_coords", "out_shape"
                                      ])

from interp_2d_iface import GridDef

def tools_to_stack_2dgrids(src_arr, srcgrid : GridDef,
                           tgtgrid : GridDef, flat_dim = None):
    import xarray as xr
    if ((srcgrid.gtyp+srcgrid.gtyp>2) and (flat_dim is None)):
        raise ValueError("flat_dim must be specified for non-rectangular grids!")
    # src is rectangular
    if (srcgrid.gtyp == 1):
        intp_dims_src = ["lat", "lon"]
    else:
        intp_dims_src = [flat_dim]
    nonintp_dims_src = [d for d in src_arr.dims if d not in intp_dims_src]
    
    nonintp_ndims_src = [len(src_arr[src_dim]) for src_dim in nonintp_dims_src]
    intp_ndims_src = [len(src_arr[src_dim]) for src_dim in intp_dims_src]
    src_dim_order = nonintp_dims_src+intp_dims_src
    src_stackshape = [max(np.prod(nonintp_ndims_src),1)]+\
    [np.prod(intp_ndims_src)]

    latattrs = {"units" : "degrees_north", "long_name":"Latitude",
    "standard_name":"latitude", "axis":"X"}
    lonattrs = {"units" : "degrees_east", "long_name":"Longitude",
    "standard_name":"longitude", "axis":"Y"}

    if (tgtgrid.gtyp == 1):
        intp_dims_tgt = ["lat", "lon"]
        intp_ndims_tgt = [len(tgtgrid.lats), len(tgtgrid.lons)]
        intp_coords_tgt = {
            "lat" : xr.DataArray(data=tgtgrid.lats, attrs=latattrs),
            "lon" : xr.DataArray(data=tgtgrid.lons, attrs=lonattrs),
        }
    else:
        intp_dims_tgt = [flat_dim]
        # Reduced
        if (tgtgrid.gtyp == 2):
            intp_ndims_tgt = [np.sum(tgtgrid.reduced_pts)]
            intp_coords_tgt = {
            "lat" : xr.DataArray(data=tgtgrid.lats, dims=["lat"], attrs=latattrs),
            "reduced_points" : xr.DataArray(data=tgtgrid.reduced_pts, dims=["lat"]),
        }
        # Unstructured 
        else:
            intp_ndims_tgt = [len(tgtgrid.lats)]
            intp_coords_tgt = {
            "lat" : xr.DataArray(data=tgtgrid.lats, dims=[flat_dim], attrs=latattrs),
            "lon" : xr.DataArray(data=tgtgrid.lons, dims=[flat_dim], attrs=lonattrs),
        }
            
    out_dim_order = nonintp_dims_src+intp_dims_tgt
    #out_stackshape = [src_stackshape[0]]+[np.prod(intp_ndims_tgt)]
    out_shape = tuple(nonintp_ndims_src+intp_ndims_tgt)

    out_coords = {
        **{dim: src_arr[dim] for dim in nonintp_dims_src},
        **intp_coords_tgt
    }
    
    tgt_ndims = nonintp_ndims_src+intp_ndims_tgt
    
    return StackTools(src_dim_order, None, out_dim_order, src_stackshape, None, out_coords, out_shape)

def tools_to_stack_xarrays(src_arr, dst_arr, intp_dim_name):
    """
    Returns all tools to reorder and reshape arrays using numpy
    """

    nonintp_dims_src = [d  for d in src_arr.dims if d != intp_dim_name]
    nonintp_dims_dst = [d  for d in dst_arr.dims if d != intp_dim_name]

    common_dim_names = list(set(nonintp_dims_src) & set(nonintp_dims_dst))
    unique_dim_names = list(set(nonintp_dims_src) ^ set(nonintp_dims_dst))
    onlysrc_dim_names = [d for d in nonintp_dims_src if d in unique_dim_names]
    onlydst_dim_names = [d for d in nonintp_dims_dst if d in unique_dim_names]

    src_dim_order = common_dim_names+onlysrc_dim_names
    dst_dim_order = common_dim_names+onlydst_dim_names
    if intp_dim_name:
        src_dim_order = src_dim_order+[intp_dim_name]
        dst_dim_order = dst_dim_order+[intp_dim_name]

    com_ndims = [len(src_arr[com_dim]) for com_dim in common_dim_names]
    src_ndims = [len(src_arr[src_dim]) for src_dim in onlysrc_dim_names]
    src_intp_ndims = [len(src_arr[intp_dim_name])] if intp_dim_name in src_arr.dims else []

    dst_ndims = [len(dst_arr[dst_dim]) for dst_dim in onlydst_dim_names]
    dst_intp_ndims = [len(dst_arr[intp_dim_name])] if intp_dim_name in dst_arr.dims else []

    src_stackshape = tuple([int(np.array(ndims).prod()) if len(ndims) > 0 else 1 for ndims in [com_ndims, src_ndims, src_intp_ndims] ])
    dst_stackshape = tuple([int(np.array(ndims).prod()) if len(ndims) > 0 else 1 for ndims in [com_ndims, dst_ndims, dst_intp_ndims]])

    out_stackshape = tuple([int(np.array(ndims).prod()) if len(ndims) > 0 else 1  for ndims in [com_ndims, src_ndims, dst_ndims, dst_intp_ndims]])
    out_shape = tuple(com_ndims+src_ndims+dst_ndims+dst_intp_ndims)
    if len(out_shape) == 0:
        out_shape = None
        out_dim_order = None
    else:
        out_dim_order =  common_dim_names+onlysrc_dim_names+onlydst_dim_names
        if intp_dim_name:
            out_dim_order = out_dim_order+[intp_dim_name]
    out_coords = {
        **{com_dim: src_arr.coords[com_dim] for com_dim in common_dim_names},
        **{src_dim: src_arr.coords[src_dim] for src_dim in onlysrc_dim_names},
        **{dst_dim: dst_arr.coords[dst_dim] for dst_dim in onlydst_dim_names},
    }
    if intp_dim_name:
        out_coords[intp_dim_name] = dst_arr.coords[intp_dim_name]

    arglist = [src_dim_order, dst_dim_order, out_dim_order]+\
    [src_stackshape, dst_stackshape, out_coords, out_shape]

    return StackTools(*arglist)
