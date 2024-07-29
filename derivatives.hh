#include "ODESolve.hh"
#include "arrays.hh"
#include "freqs_ntT.hh"

class derivatives : public ODESolve<freqs_ntT>
{
private:
    double num_bins;
    double a_start;
    double a_end;
    double sterile_mass;
    double mixing_angle;

public:
    derivatives(int, double, double, double, double, dummy_vars*, double, double, double);
    
    double get_sterile_mass();
    double get_mixing_angle();
    double get_a_end();
    double get_a_start();
    void set_sterile_mass(double);
    void set_mixing_angle(double);
    void set_a_end(double);
    void set_a_start(double);

    void f(double, freqs_ntT*, freqs_ntT*);
};

