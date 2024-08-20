#!/usr/bin/bash

mass="290"
lifetime="1.00"
foldername="decay_mass-290_life-1000"

if [ -d $foldername ]; then
    if [ ! -z "$( ls -A $foldername )" ]; then
        echo "$foldername already exists. Stopping execution."
        echo "Rename the folder or rename variable foldername in this script."
        exit 1
    else
        echo "$foldername already exists, but is empty. Output will be placed in $foldername"
    fi
else
    echo "Creating Directory $foldername"
    mkdir $foldername
fi

set -x

rm decay
g++ run_decays.cc array_methods.cc decays.cc decays_only.cc freqs_ntT.cc thermodynamics.cc -std=c++11 -o decay

./decay $mass $lifetime $foldername