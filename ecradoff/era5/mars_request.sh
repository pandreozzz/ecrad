#!/bin/bash

SWITCH=$1
DATE="1956-10-28"
LEVLIST=$(seq -s "/" 1 137)
STEPS=00/06/12/18
GRID="O96" #"0.25/0.25"
griddesc="O96" #"025deg"

#if [[ ${SWITCH} == 0 ]]
#then
mars << EOF
retrieve,
class=ea,
date=${DATE},
expver=1,
levelist=${LEVLIST},
levtype=ml,
param=75/76/130/133/152/203/246/247/248,
step=${STEPS},
stream=oper,
time=06:00:00,
type=fc,
grid=${GRID},
target="./data/era5_${DATE}_${griddesc}_ml.grib"
EOF

#else sfc fields
mars << EOF
retrieve,
class=ea,
date=${DATE},
expver=1,
levtype=sfc,
param=164.128/165.128/166.128/172.128/178.128/208.128/212.128/235.128/243.128,
step=${STEPS},
stream=oper,
time=06:00:00,
type=fc,
grid=${GRID},
target="./data/era5_${DATE}_${griddesc}_sfc.grib"
EOF
#fi
