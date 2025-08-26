import os
import ctypes as ct
import numpy as np

f_lib = ct.CDLL('./f_src/fvertintp.so')
c_real = ct.c_double
c_real_ptr = np.ctypeslib.ndpointer(c_real)#ct.POINTER(c_real)
c_int  = ct.c_int32
c_int_ptr = np.ctypeslib.ndpointer(c_int)#ct.POINTER(c_int)

f_lib.interp.argtypes = [c_real_ptr,]*2+[c_int_ptr, c_real_ptr]+[c_int,]*6
f_lib.interp_fld.argtypes = [c_real_ptr,]*2+[c_int_ptr, c_real_ptr]+[c_int,]*6

def interp(psrc, ptgt, chunk_size_max=1000):
    """
        Interpolates psrc to ptgt
        psrc(ncom,nsrc,nlevsrc)
        ptgt(ncom,ntgt,nlevtgt)

        returns
        tgtlevs(ncom,nsrc,ntgt,nlevtgt), weights(ncom,nsrc,ntgt,nlevtgt)
    """
    ncom,ntgt,nlevtgt  = ptgt.shape
    ncom2,nsrc,nlevsrc = psrc.shape


    if (ncom != ncom2):
        raise ValueError(
                "Different common dimension size between source and target!"+\
                f"({ncom2},{nsrc}) and ({ncom},{nsrc})")


    outshape = (ncom,nsrc,ntgt,nlevtgt)

    tgtlevs = np.zeros(outshape, dtype=c_int)
    weights = np.zeros(outshape, dtype=c_real)

    # Enforce contiguity
    psrc_cont = np.ascontiguousarray(psrc)
    ptgt_cont = np.ascontiguousarray(ptgt)
    tgtlevs = np.ascontiguousarray(tgtlevs)
    weights = np.ascontiguousarray(weights)

    f_lib.interp(
        psrc_cont.astype(c_real), ptgt_cont.astype(c_real),
        tgtlevs, weights,
        c_int(ncom), c_int(nsrc), c_int(ntgt),
        c_int(nlevsrc), c_int(nlevtgt), c_int(chunk_size_max)
        )
    del psrc_cont, ptgt_cont

    return tgtlevs, weights

def interp_fld(fsrc, tgtlevs, weights, chunk_size_max=1000):
    """
        Interpolates fields according to weights
        fsrc(ncom,nsrc,nlevsrc)
        tgtlevs(ncom,ntgt,nlevtgt)
        weights(ncom,ntgt,nlevtgt)

        returns
        fdst(ncom,nsrc,ntgt,nlevtgt)
    """
    ncom,nsrc,nlevsrc = fsrc.shape

    ncom2,ntgt,nlevtgt   = tgtlevs.shape
    ncom3,ntgt2,nlevtgt2 = weights.shape

    if (ncom != ncom2) or (ncom2 != ncom3) or (ntgt != ntgt2) or (nlevtgt != nlevtgt2):
        raise ValueError(
                "Some dimensions are incompatible!!"+\
                f"fsrc(ncom={ncom},nsrc={nsrc},nlevsrc{nlevsrc})"+\
                f"tgtlevs(ncom={ncom2},ntgt={ntgt},nlevtgt{nlevtgt})"+\
                f"tgtlevs(ncom={ncom3},ntgt={ntgt2},nlevtgt{nlevtgt2})")

    fdstshape = (ncom,nsrc,ntgt,nlevtgt)
    fdst = np.zeros(fdstshape, dtype=c_real)

    # Enforce contiguity
    fsrc_cont = np.ascontiguousarray(fsrc)
    fdst = np.ascontiguousarray(fdst)
    tgtlevs_cont = np.ascontiguousarray(tgtlevs)
    weights_cont = np.ascontiguousarray(weights)

    f_lib.interp_fld(
            fsrc_cont.astype(c_real),
            fdst,
            tgtlevs_cont.astype(c_int),
            weights_cont.astype(c_real),
            c_int(ncom), c_int(nsrc), c_int(ntgt),
            c_int(nlevsrc), c_int(nlevtgt), c_int(chunk_size_max)
            )
    del fsrc_cont, tgtlevs_cont, weights_cont

    return fdst

if __name__=="__main__":
    print(f"Hi! This is the interface to the fortran pressure interpolation routine")
