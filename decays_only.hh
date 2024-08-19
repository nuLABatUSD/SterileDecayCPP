#include "ODESolve.hh"
#include "arrays.hh"
#include "freqs_ntT.hh"

class derivatives : public ODESolve<freqs_ntT>
{
private:
    double num_bins;
    double E_low;
    double E_high;
    double a_start;
    double a_end;
    double sterile_mass;
    double mixing_angle;

public:
    derivatives(int, double, double, double, double, double, double, dummy_vars*, double, double, double);
    ~derivatives();

    double get_sterile_mass();
    double get_mixing_angle();
    double get_low();
    double get_high();
    double get_a_end();
    double get_a_start();
    freqs_ntT* get_yvalues();

    void set_sterile_mass(double);
    void set_mixing_angle(double);
    void set_low(double);
    void set_high(double);
    void set_a_end(double);
    void set_a_start(double);

    dummy_vars* retrieve_separations();
    void update(double, double);
    void shift_x();

    void f(double, freqs_ntT*, freqs_ntT*);
    
    void print_eps_file(const std::string&);
    bool inch_forward(double, double a1, std::string& folder_name, std::string& file_name, bool = true);
    
    double calc_Neff();
};

