#!/usr/bin/bash

set -x

rm decayMPI

mpic++ run_MPI.cc derivativesMPI.cc array_methods.cc decays.cc decays_only.cc freqs_ntT.cc thermodynamics.cc -std=c++11 -o decayMPI

mpiexec -n 4 decayMPI

