#include "ODESolve.hh"
#include "arrays.hh"
#include "freqs_ntT.hh"

class derivatives : public ODESolve<freqs_ntT>
{
private:
    double sterile_mass;
    double mixing_angle;
    double a_end;

public:
    derivatives(double, double, double);
    
    double get_sterile_mass();
    double get_mixing_angle();
    double get_a_end();
    void set_sterile_mass(double);
    void set_mixing_angle(double);
    void set_a_end(double);

    int get_bins();
    void f(double, freqs_ntT*, freqs_ntT*);
};

