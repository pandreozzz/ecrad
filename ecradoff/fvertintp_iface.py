import ctypes as ct
import numpy as np

f_lib = ct.CDLL('./f_src/fvertintp.so')
c_real = ct.c_double
c_real_ptr = np.ctypeslib.ndpointer(c_real)#ct.POINTER(c_real)
c_int  = ct.c_int32
c_int_ptr = np.ctypeslib.ndpointer(c_int)#ct.POINTER(c_int)

f_lib.interp.argtypes = [c_real_ptr,]*2+[c_int_ptr, c_real_ptr]+[c_int,]*4
f_lib.interp_fld.argtypes = [c_real_ptr,]*2+[c_int_ptr, c_real_ptr]+[c_int,]*4
f_lib.interp_multi_fld.argtypes = [c_real_ptr,]*2+[c_int_ptr, c_real_ptr]+[c_int,]*5

def interp(psrc, ptgt):
    """
        Interpolates psrc to ptgt
        psrc(nx,ny,nlevsrc)
        ptgt(nx,ny,nlevtgt)

        returns
        tgtlevs(nx,ny,nlevtgt), weights(nx,ny,nlevtgt)
    """
    nx,ny,nlevtgt   = ptgt.shape
    nx2,ny2,nlevsrc = psrc.shape



    if (nx != nx2) or (ny != ny2):
        raise ValueError(
                "Different horizontal dimensions between source and target!"+\
                f"({nx},{ny}) and ({nx2},{ny2})")


    tgtlevs = np.zeros_like(ptgt, dtype=c_int)
    weights = np.zeros_like(ptgt, dtype=c_real)

    # Enforce contiguity
    psrc = np.ascontiguousarray(psrc)
    ptgt = np.ascontiguousarray(ptgt)
    tgtlevs = np.ascontiguousarray(tgtlevs)
    weights = np.ascontiguousarray(weights)

    f_lib.interp(
            psrc.astype(c_real),
            ptgt.astype(c_real),
            tgtlevs,
            weights,
            c_int(nx), c_int(ny),
            c_int(nlevsrc), c_int(nlevtgt)
            )
    return tgtlevs, weights

def interp_fld(fsrc, tgtlevs, weights):
    """
        Interpolates fields according to weights
        fsrc(nfields,nx,ny,nlevsrc)
        tgtlevs(nx,ny,nlevdst)
        weights(nx,ny,nlevdst)

        returns
        fdst(nx,ny,nlevdst)
    """
    fsrc_shape = fsrc.shape
    if len(fsrc_shape) == 4:
        nflds,nx2,ny2,nlevsrc = fsrc_shape
    else:
        nflds = 0
        nx2,ny2,nlevsrc  = fsrc_shape
    nx,ny,nlevdst    = tgtlevs.shape
    nx1,ny1,nlevdst1 = weights.shape

    if (nx != nx1) or (ny != ny1) or (nlevdst != nlevdst1):
        raise ValueError(
                "Different shape dimensions between tgtlevs and weights!"+\
                f"({nx},{ny},{nlevdst}) and ({nx1},{ny1},{nlevdst1})")
    if (nx != nx2) or (ny != ny2):
        raise ValueError(
                "Different horizontal dimensions between weights and source fields!"+\
                f"({nx},{ny}) and ({nx2},{ny2})")

    fdst = np.zeros_like(weights, dtype=c_real)
    if nflds > 0:
        fdst = np.concatenate([fdst[None, ...],]*nflds, axis=0)

    # Enforce contiguity
    fsrc = np.ascontiguousarray(fsrc)
    fdst = np.ascontiguousarray(fdst)
    tgtlevs = np.ascontiguousarray(tgtlevs)
    weights = np.ascontiguousarray(weights)

    if nflds == 0:
        f_lib.interp_fld(
                fsrc.astype(c_real),
                fdst,
                tgtlevs.astype(c_int),
                weights.astype(c_real),
                c_int(nx), c_int(ny),
                c_int(nlevsrc), c_int(nlevdst)
                )
    else:
        f_lib.interp_multi_fld(
                fsrc.astype(c_real),
                fdst,
                tgtlevs.astype(c_int),
                weights.astype(c_real),
                c_int(nflds),
                c_int(nx), c_int(ny),
                c_int(nlevsrc), c_int(nlevdst)
                )

    return fdst

if __name__=="__main__":
    from time import time
    import os
    import xarray as xr
    import numpy as np

    renamedic=dict(longitude="lon", latitude="lat", level="lev")
    cams = {vers : xr.open_mfdataset(f"cams_{vers}_mini.nc", parallel=True).rename(renamedic) for vers in ("43r3","49r2")}
    ifs_lnsp = xr.open_mfdataset(f"mini_lnsp.nc", parallel=True)

    vers = "43r3"
    # src = "60"
    # tgt = "137"

    stdcoeffs=xr.open_dataset("stdcoeffs.nc")

    cams_sel = cams[vers].sortby("lat", ascending=False).drop_vars("leadtime")

    p_src = cams_sel[PDIM]
    p_tgt = (stdcoeffs["coeff_137"].sel(order=0, drop=True) +\
             stdcoeffs["coeff_137"].sel(order=1,drop=True)*np.exp(ifs_lnsp["lnsp"])).rename(lev_137="lev")

    this_p_src = p_src.transpose("lat","lon","lev").isel(lat=[0,], lon=[0,])
    this_p_tgt = p_tgt.transpose("time","lat","lon","lev").isel(lat=[0,], lon=[0,])
    src_hordim = this_p_src.isel(lev=0,drop=True).dims
    tgt_hordim = this_p_tgt.isel(lev=0,drop=True).dims

    tgtlevs1 = xr.zeros_like(this_p_tgt).astype(int)
    weights1 = xr.zeros_like(this_p_tgt)

    start = time()
    for t in this_p_tgt.time:
        l, w = interp(psrc=this_p_src.values,ptgt=this_p_tgt.sel(time=t).values)
        tgtlevs1.loc[tgtlevs1.time==t] = l
        weights1.loc[weights1.time==t] = w

    print(f"It took {time() - start}s")
