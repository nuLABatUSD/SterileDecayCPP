#include "ODESolve.hh"
#include "arrays.hh"
#include "freqs_ntT.hh"
#include "decays_only.hh"
#include "constants.hh"
#include "decays.h"
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>


using std::cout;
using std::endl;
using std::to_string;
using std::string;
using std::ofstream;




int main(int argc, char* argv[]){
    /*
    * These four terms and E_low and E_high are constrained in order for our dummy_vars object to function as intended. In particular, we 
    * must obey the inequality scales_num >= floor[log( a_end / a_start ) / log(E_low * (num - 11) / E_high) + 2]
    */
    if (argc != 4)
    {
        cout << "Requires THREE inputs: ms_MeV lifetime_seconds output_folder_name" << endl << "Abort." << endl;
        return 1;
    }
    
    double ms = std::atof(argv[1]);
    double tau_s = std::atof(argv[2]);
    string folder = std::string(argv[3]);
    
    double theta = 1.e-6 * sqrt(get_lifetime(ms, 1.e-6)*_hbar_/tau_s);

    double a_start = 0.1;
    double a_end = 10.00;
    int num = 51;
    int scale_num = 9;
    
   // double ms = 300;
   // double theta = 1.22e-5;
    double time = 0;
    double temp = 1 / a_start;

    double E_low = min_low(ms);
    double E_high = (ms + 10) / 2;
    

    dummy_vars* scales = new dummy_vars(scale_num);
    seek_as(a_start, a_end, scales);

    // first evaluation done separately for setup
    double a_low = scales->get_value(0);
    double a_high = scales->get_value(1);
    gel_linspace_gl* eps = new gel_linspace_gl(E_low * a_low, E_high * a_high, num);
    dummy_vars* freqs = new dummy_vars(6 * num);
    for(int i = 0; i < num; i++){
        double E = eps->get_value(i);
        double f = 1 / (exp(E) + 1);
        freqs->set_value(i, f);
        freqs->set_value(i + num, f);
        freqs->set_value(i + 2 * num, f);
        freqs->set_value(i + 3 * num, f);
        freqs->set_value(i + 4 * num, f);
        freqs->set_value(i + 5 * num, f);
    }

    double ns =  (3 * _zeta_3_ / (2 * pow(_PI_,2))) * _gwd_ * pow(10,3) / _gsdec_;
    freqs_ntT* input = new freqs_ntT(num, E_low, E_high, a_low, a_high, ms, theta, freqs, ns, time, temp);
    derivatives* sim = new derivatives(num, E_low, E_high, a_low, a_high, ms, theta, freqs, ns, time, temp);

    
    sim->set_ics(a_start, input, 0.01 * a_start);
    
    string filename = "decay_only-";
    
    cout << "Solving DECAYS-ONLY with m_s = " << ms << " MeV and lifetime " << tau_s << "s from T_cm = " << 1/scales->get_value(0) << " MeV to T_cm = " << 1/scales->get_value(scale_num-1) << " MeV" << endl;
    
    for (int i = 0; i < scale_num -1; i++)
    {
        
        string fn = filename + to_string(i);
        sim->inch_forward(scales->get_value(i), scales->get_value(i+1), folder, fn, true);
        
    }
    
    cout << "N_eff = " << sim->calc_Neff() << endl;
    
    string results_name = folder + "/results.txt";
    ofstream results(results_name);
    results << "m_s = " << ms << endl;
    results << "theta = " << theta << endl;
    results << "N_eff = " << sim->calc_Neff() << endl;
    results.close();
  
    delete sim;
    delete input;
    delete eps;
    delete freqs;
    delete scales;
    
    return 0;
}
