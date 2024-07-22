#ifndef _DECAYS_H_
#define _DECAYS_H_
#include "arrays.hh"

double get_rate(double, double, int);
double get_lifetime(double, double);
double get_monoenergy(double, double, double);
void compute_kinetics(double, double, double, double*, double*, double*);
double get_gamma_a(double);
double get_gamma_b(double);
double get_decay_type_one(double, double, double, double, double);
double get_decay_type_two(double, double, double, double);
double get_decay_type_three(double, double, double);
double type_four_integrand(double, double);
double get_decay_type_four(double, double, double, double);
void compute_dPdtdE(linspace_and_gl*, double, double, double, dummy_vars*, dummy_vars*, dummy_vars*, dummy_vars*, dummy_vars*, dummy_vars*);
void compute_full_term(linspace_and_gl*, double, double, double, dummy_vars*, dummy_vars*, dummy_vars*, dummy_vars*, dummy_vars*, dummy_vars*);
#endif