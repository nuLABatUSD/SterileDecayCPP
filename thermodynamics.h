#ifndef _THERMODYNAMICS_H_
#define _THERMODYNAMICS_H_

double energy_f(double, double, double);
double pressure_f(double, double, double);
double energy_temp_f(double, double, double);
double pressure_temp_f(double, double, double);
void energy_and_pressure(double, double, double*, double*);
void energy_pressure_and_derivs(double m, double T, double* rho, double* P, double* rho_temp, double* P_temp);
#endif