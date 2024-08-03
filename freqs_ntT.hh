#ifndef _FREQS_NTT_HH
#define _FREQS_NTT_HH
#include "arrays.hh"

class freqs_ntT : public dep_vars
{
    protected:
    int num_bins;
    double E_low;
    double E_high;
    double a_start;
    double a_end;
    gel_linspace_gl* eps;
    double sterile_mass;
    double mixing_angle;

    public:

    freqs_ntT(int, double, double, double, double, double, double, dummy_vars*, double, double, double);
    freqs_ntT(freqs_ntT*);
    ~freqs_ntT();

    void eps_shift(double, double);
    double get_eps_value(int);
    gel_linspace_gl* get_eps();
    double get_ms();
    double get_theta();
    int get_num_bins();
    double get_low();
    double get_high();
    double get_a_start();
    double get_a_end();

    void set_low(double);
    void set_high(double);
    void set_a_start(double);
    void set_a_end(double);
    void set_ms(double);
    void set_theta(double);

    double get_temp();
    void set_temp(double);
    double get_time();
    void set_time(double);
    double get_ns();
    void set_ns(double);

    double get_sterile_density();
    double get_photon_density();
    double get_ind_neutrino_density(int, double);
    double get_neutrino_density(double);
    double get_dtda(double);
    double get_dnsda(double);
    void compute_dfda(double, dummy_vars*, dummy_vars*, dummy_vars*, dummy_vars*, dummy_vars*, dummy_vars*);
    void compute_dfda_cube(double, dep_vars*);
    double get_dQda(double);
    double get_dTda(double);
    void compute_derivs(double, dummy_vars*, dummy_vars*, dummy_vars*, dummy_vars*, dummy_vars*, dummy_vars*, double*, double*, double*);

    dummy_vars* get_separations();
};



#endif