#include "ODESolve.hh"
#include "freqs_ntT.hh"
#include "arrays.hh"
#include "decays_only.hh"

class derivativesMPI 
{
    protected:
        double x_value;
        freqs_ntT* y_values;
        double dx_value;
        
        int total_ODE_steps;
        int total_ODE_rejected_steps;
        
        int myid;
        int numprocs;
        
        int num_bins;
        double E_low;
        double E_high;
        double a_start;
        double a_end;
        double sterile_mass;
        double mixing_angle;
        
        derivatives* just_decays;
        integration** nu_nu_coll_objects;
        
        double** nu_nu_tolerances;

        
    public:
        derivativesMPI(int, double, double, double, double, double, double, dummy_vars*, double, double, double);
        ~derivativesMPI();
//        derivativesMPI(int, int, int, double, double, double, double, double, double, dummy_vars*, double, double, double);
        
        void set_ics(double, freqs_ntT*, double);
        void print_state();
        void print_csv(ostream&);
        
        void f(double, freqs_ntT*, freqs_ntT*);
        double first_derivative(double, freqs_ntT*, freqs_ntT*, double);
        void f_evaluate(freqs_ntT*);
        
        void RKCash_Karp(double, freqs_ntT*, double, double*, freqs_ntT*, freqs_ntT*);
        bool step_accept(freqs_ntT*, freqs_ntT*, freqs_ntT*, double, double*, bool=false, bool=false);
        
        bool RKCK_step(double, freqs_ntT*, double, double*, freqs_ntT*, double*);
        bool RKCK_step_advance();
        bool ODEOneRun(int N_step, int dN, double x_final, const std::string& file_name, bool verbose = false, bool print_csv_file = true);
        
        bool run(int N_step, int dN, double x_final, const std::string& file_name, bool verbose = false);     
       
    
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

        void print_eps_file(const std::string&);
        bool inch_forward(double, double a1, std::string& folder_name, std::string& file_name, bool = true);
        
        double calc_Neff();
};
