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

class integration
{
    protected:
    gel_linspace_gl* eps;
    int p1;
    int count;
    dep_vars* outer_vals;
    dep_vars** inner_vals;
    dummy_vars** p3_vals;
    double*** Fvv_values;
    double*** Fvvbar_values;

    double temp_cm;
    double me_scaled;
    dummy_vars* p2_vals;


    public:
    integration(gel_linspace_gl*, int);
    integration(integration*);
    ~integration();
    dummy_vars** get_p3();
    double Fvv_comp(freqs_ntT*, bool, int, int, int, double);
    void populate_Fvv(freqs_ntT*, double);
    double Fvvbar_comp(freqs_ntT*, bool, int, int, int, double);
    void populate_Fvvbar(freqs_ntT*, double);
    double J1(double, double, double);
    double J2(double, double);
    double J3(double, double, double);
    double K1(double, double);
    double K2(double, double, double);
    double K3(double, double, double);
    
    double interior_integral(int, int);
    void whole_integral(freqs_ntT*, double, double, double*);
};

class nu_e_collision_R1
{
    protected:
    int p1;
    int count;
    int* p4_lows;
    int* p4_highs;
    double temp_cm;
    double me_scaled;
    gel_linspace_gl* eps;
    dep_vars* outer_vals;
    dep_vars** inner_vals;
    dummy_vars* q2_vals;
    dummy_vars** q3_vals;
    double*** F_values;
    
    public:
    nu_e_collision_R1(gel_linspace_gl*, int, double);
    ~nu_e_collision_R1();

    double get_temp_cm();
    dummy_vars** get_q3();
    double M1_R1(double, double, int);
    double M2_R1(double, double, int);
    double F_comp(freqs_ntT*, int, int, int, int);
    void populate_F(freqs_ntT*, int);
    void print_F(int);
    double interior_integral_R1(int, int);
    void whole_integral(freqs_ntT*, double, double*);
    
};

class nu_e_collision_R2
{
    protected:
    gel_linspace_gl* eps;
    int p1;
    double p1_energy;
    double scaled_me;
    double p1_me;
    double temp_cm;
    
    double q_cut_1;
    double q_cut_2;
    double q_cut_3;
    dep_vars* q_trans_2;
    dep_vars* q_lim_1;
    int* p4_lows;
    int* p4_highs;

    dummy_vars* q3_vals;
    dep_vars* outer_vals;
    dummy_vars** q2_vals;
    dep_vars** inner_vals;
    double*** F_values;
    
    
    public:
    nu_e_collision_R2(gel_linspace_gl*, int, double);
    double F_comp(freqs_ntT*, int, int, int, int);
    void populate_F(freqs_ntT*, int);
    double integrated_M_1_prime(double, double, int);
    double M_1_1(double, double, double, int);
    double M_1_2(double, double, int);
    double M_1_3(double, double, double, double, int);
    double M_1_4(double, double, double, double, int);
    double integrated_M_2_prime(double, double, int);
    double M_2_1(double, double, double, int);
    double M_2_2(double, double, int);
    double M_2_3(double, double, double, double, int);
    double M_2_4(double, double, double, double, int);
    double inner_integral(int, int);
    double whole_integral(freqs_ntT*, double, double, double*);
    
    ~nu_e_collision_R2();
};
#endif