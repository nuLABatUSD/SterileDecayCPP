#ifndef _FREQS_NTT_HH
#define _FREQS_NTT_HH
#include "arrays.hh"

class freqs_ntT : public dep_vars
{
    protected:
    int num_bins;
    double a_start;
    double a_end;
    linspace_and_gl* eps;
    double sterile_mass;
    double mixing_angle;

    public:

    freqs_ntT(int, double, double, double, double, linspace_and_gl*, dummy_vars*, double, double, double);
    freqs_ntT(int, double, double);
    ~freqs_ntT();

    double get_eps_value(int);
    linspace_and_gl* get_eps();
    double get_ms();
    double get_theta();
    int get_num_bins();

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

    dummy_vars* get_separations();
};

#endif