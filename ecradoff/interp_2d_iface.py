import os
import ctypes as ct
import numpy as np

f_lib = ct.CDLL('./f_src/interp_2d.so')
c_real = ct.c_double
c_real_ptr = np.ctypeslib.ndpointer(c_real)#ct.POINTER(c_real)
c_int  = ct.c_int32
c_int_ptr = np.ctypeslib.ndpointer(c_int)#ct.POINTER(c_int)

f_lib.interp_2d_red2red.argtypes = [c_real_ptr]*6+[c_int]*7+[c_int_ptr]*2+[c_int]+[c_real]
f_lib.interp_2d_red2rec.argtypes = [c_real_ptr]*6+[c_int]*7+[c_int_ptr]+[c_int]+[c_real]
f_lib.interp_2d_rec2red.argtypes = [c_real_ptr]*6+[c_int]*7+[c_int_ptr]+[c_int]+[c_real]
f_lib.interp_2d_rec2rec.argtypes = [c_real_ptr]*6+[c_int]*7+[c_int]+[c_real]

class GridDef:
    def __init__(self, name : str, gtyp : int,
                 lons : np.ndarray, lats : np.ndarray,
                 reduced_pts : np.ndarray = None
                 ):
        self.name = name
        if gtyp not in [1,2,3]:
            raise ValueError(f"Unrecognized grid type {gtyp}")
        self.gtyp = gtyp
        self.lons = lons
        self.lats = lats
        if gtyp == 2:
            self.reduced_pts = reduced_pts
        
        if (gtyp == 1):
            self.npts = len(lons)*len(lats)
        elif (gtyp == 2):
            if (len(lats) != len(lons)) or (len(lats) != len(reduced_pts)):
                raise ValueError(
                    f"For grid type reduced (gtyp={gtyp}) "+\
                    f"lons ({len(lons)}), lats ({len(lats)}), reduced_pts ({len(reduced_pts)}) "+\
                    "must have the same length!")
            self.npts = reduced_pts.sum().astype(int)
        else:
            if (len(lats) != len(lons)):
                raise ValueError(
                    f"For grid type unstructured (gtyp={gtyp}) "+\
                    f"lons ({len(lons)}) and lats ({len(lats)}) "+\
                    "must have the same length!")
            self.npts = len(lons)

def interp_2d(fsrc, srcgrid : GridDef, tgtgrid : GridDef,
              chunk_size_max=1000, auto_chunk_size=True, abs_tolerance=1.e-3):
    """
    """

    nflds, nxysrc = fsrc.shape
    if (nxysrc != srcgrid.npts):
        raise ValueError(f"Incompatible number of gridpoints for fsrc ({nxysrc}) and grid specs ({srcgrid.npts})!")
    
    outshape = (nflds, tgtgrid.npts)

    # Enforce contiguity
    fsrc_cont = np.ascontiguousarray(fsrc)
    fdst = np.ascontiguousarray(np.zeros(outshape, dtype=c_real))

    if (tgtgrid.gtyp == 1):
        if auto_chunk_size:
            chunk_size_max = 1
        if (srcgrid.gtyp == 1):
            f_lib.interp_2d_rec2rec(
                fsrc_cont.astype(c_real), fdst,
                np.ascontiguousarray(srcgrid.lats).astype(c_real),
                np.ascontiguousarray(srcgrid.lons).astype(c_real),
                np.ascontiguousarray(tgtgrid.lats).astype(c_real),
                np.ascontiguousarray(tgtgrid.lons).astype(c_real),
                c_int(nflds),
                c_int(len(srcgrid.lats)), c_int(len(srcgrid.lons)), c_int(srcgrid.npts), 
                c_int(len(tgtgrid.lats)), c_int(len(tgtgrid.lons)), c_int(tgtgrid.npts), 
                c_int(chunk_size_max), c_real(abs_tolerance)
                )
        if (srcgrid.gtyp == 2):
            f_lib.interp_2d_red2rec(
                fsrc_cont.astype(c_real), fdst,
                np.ascontiguousarray(srcgrid.lats).astype(c_real),
                np.ascontiguousarray(srcgrid.lons).astype(c_real),
                np.ascontiguousarray(tgtgrid.lats).astype(c_real),
                np.ascontiguousarray(tgtgrid.lons).astype(c_real),
                c_int(nflds),
                c_int(len(srcgrid.lats)), c_int(len(srcgrid.lons)), c_int(srcgrid.npts), 
                c_int(len(tgtgrid.lats)), c_int(len(tgtgrid.lons)), c_int(tgtgrid.npts),
                np.ascontiguousarray(srcgrid.reduced_pts).astype(c_int),
                c_int(chunk_size_max), c_real(abs_tolerance)
                )
    elif (tgtgrid.gtyp == 2):
        if auto_chunk_size:
            chunk_size_max = 1
        if (srcgrid.gtyp == 1):
            f_lib.interp_2d_rec2red(
                fsrc_cont.astype(c_real), fdst,
                np.ascontiguousarray(srcgrid.lats).astype(c_real),
                np.ascontiguousarray(srcgrid.lons).astype(c_real),
                np.ascontiguousarray(tgtgrid.lats).astype(c_real),
                np.ascontiguousarray(tgtgrid.lons).astype(c_real),
                c_int(nflds),
                c_int(len(srcgrid.lats)), c_int(len(srcgrid.lons)), c_int(srcgrid.npts), 
                c_int(len(tgtgrid.lats)), c_int(len(tgtgrid.lons)), c_int(tgtgrid.npts), 
                np.ascontiguousarray(tgtgrid.reduced_pts).astype(c_int),
                c_int(chunk_size_max), c_real(abs_tolerance)
                )
        if (srcgrid.gtyp == 2):
            f_lib.interp_2d_red2red(
                fsrc_cont.astype(c_real), fdst,
                np.ascontiguousarray(srcgrid.lats).astype(c_real),
                np.ascontiguousarray(srcgrid.lons).astype(c_real),
                np.ascontiguousarray(tgtgrid.lats).astype(c_real),
                np.ascontiguousarray(tgtgrid.lons).astype(c_real),
                c_int(nflds),
                c_int(len(srcgrid.lats)), c_int(len(srcgrid.lons)), c_int(srcgrid.npts), 
                c_int(len(tgtgrid.lats)), c_int(len(tgtgrid.lons)), c_int(tgtgrid.npts),
                np.ascontiguousarray(srcgrid.reduced_pts).astype(c_int),
                np.ascontiguousarray(tgtgrid.reduced_pts).astype(c_int),
                c_int(chunk_size_max), c_real(abs_tolerance)
                )

    return fdst

if __name__=="__main__":
    print(f"Hi! This is the interface to the fortran grid interpolation routine")
