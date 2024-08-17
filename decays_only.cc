#include "ODESolve_new.hh"
#include "arrays.hh"
#include "freqs_ntT.hh"
#include "decays_only.hh"
#include "constants.hh"
#include "decays.h"
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>

using std::cout;
using std::endl;
using std::to_string;
using std::string;
using std::ofstream;

using namespace std::chrono;


derivatives::derivatives(int num, double low, double high, double start, double end, double ms, double theta, dummy_vars* freqs, double ns, double time, double temp):ODESolve(){
    num_bins = num;
    E_low = low;
    E_high = high;
    a_start = start;
    a_end = end;
    sterile_mass = ms;
    mixing_angle = theta;

    y_values = new freqs_ntT(num, E_low, E_high, a_start, a_end, sterile_mass, mixing_angle, freqs, ns, time, temp);
}

derivatives::~derivatives(){
    ; //delete y_values;
}

dummy_vars* derivatives::retrieve_separations(){
    return y_values->get_separations();
}

void derivatives::update(double new_a_start, double new_a_end){
    if (a_start != new_a_start || a_end != new_a_end)
    {
        a_start = new_a_start;
        a_end = new_a_end;
        y_values->eps_shift(new_a_start, new_a_end);
    }
}

void derivatives::shift_x()
{
    x_value = x_value + 1e-14 * 0;
}

double derivatives::get_sterile_mass(){
    return sterile_mass;
}

void derivatives::set_sterile_mass(double ms){
    sterile_mass = ms;
}

double derivatives::get_mixing_angle(){
    return mixing_angle;
}

void derivatives::set_mixing_angle(double theta){
    mixing_angle = theta;
}

double derivatives::get_low(){
    return E_low;
}

double derivatives::get_high(){
    return E_high;
}

void derivatives::set_low(double new_low){
    E_low = new_low;
}

void derivatives::set_high(double new_high){
    E_high = new_high;
}

double derivatives::get_a_end(){
    return a_end;
}

void derivatives::set_a_end(double end){
    a_end = end;
}

double derivatives::get_a_start(){
    return a_start;
}

void derivatives::set_a_start(double start){
    a_start = start;
}

void derivatives::f(double a, freqs_ntT* inputs, freqs_ntT* derivs){
    double num_bins = inputs->get_num_bins();
    dummy_vars* electron = new dummy_vars(num_bins);
    dummy_vars* anti_electron = new dummy_vars(num_bins);
    dummy_vars* muon = new dummy_vars(num_bins);
    dummy_vars* anti_muon = new dummy_vars(num_bins);
    dummy_vars* tau = new dummy_vars(num_bins);
    dummy_vars* anti_tau = new dummy_vars(num_bins);
    double* dnsda = new double;
    double* dtda = new double;
    double* dTda = new double;

    inputs->compute_derivs(a, electron, anti_electron, muon, anti_muon, tau, anti_tau, dnsda, dtda, dTda);
    for(int i = 0; i < num_bins; i++){
        derivs->set_value(i, electron->get_value(i));
        derivs->set_value(i + num_bins, anti_electron->get_value(i));
        derivs->set_value(i + 2 * num_bins, muon->get_value(i));
        derivs->set_value(i + 3 * num_bins, anti_muon->get_value(i));
        derivs->set_value(i + 4 * num_bins, tau->get_value(i));
        derivs->set_value(i + 5 * num_bins, anti_tau->get_value(i));
    }
    //file.precision(std::numeric_limits<double>::max_digits10);
    //file << a << ", " << derivs->get_value(2) << endl;
    derivs->set_ns(*dnsda);
    derivs->set_time(*dtda);
    derivs->set_temp(*dTda);
    
    /*
    inputs->compute_dfda(a, electron, anti_electron, muon, anti_muon, tau, anti_tau);
    for(int i = 0; i < num_bins; i++){
        derivs->set_value(i, electron->get_value(i));
        derivs->set_value(i + num_bins, anti_electron->get_value(i));
        derivs->set_value(i + 2 * num_bins, muon->get_value(i));
        derivs->set_value(i + 3 * num_bins, anti_muon->get_value(i));
        derivs->set_value(i + 4 * num_bins, tau->get_value(i));
        derivs->set_value(i + 5 * num_bins, anti_tau->get_value(i));
    }

    cout << "Here!" << endl;
    double dnsda = inputs->get_dnsda(a);
    cout << "Here!" << endl;
    double dtda = inputs->get_dtda(a);
    cout << "Here!" << endl;
    double dTda = inputs->get_dTda(a);
    derivs->set_ns(dnsda);
    derivs->set_time(dtda);
    derivs->set_temp(dTda);
    */
    delete electron;
    delete anti_electron;
    delete muon;
    delete anti_muon;
    delete tau;
    delete anti_tau;
    delete dnsda;
    delete dtda;
    delete dTda;

}

void derivatives::print_eps_file(const std::string& file_name)
{
    ofstream os(file_name);

    os.precision(std::numeric_limits<double>::max_digits10 - 1);
    for (int i = 0; i < num_bins-1; i++)
    {
        os << y_values->get_eps_value(i) << ", ";
    }
    os << y_values->get_eps_value(num_bins-1) << endl;
    
    os.close();
}

bool derivatives::inch_forward(double a0, double a1, string& folder_name, string& file_name, bool verbose)
{
    if (x_value != a0)
    {
        cout << "ERROR: The initial a must be equal to a_low" << endl;
        return false;
    }
    
    update(a0, a1);
    dummy_vars* a_separations = retrieve_separations();
    dx_value *= 0.01;
    
    cout << a_separations->get_value(0) << " " << a_separations->get_value(a_separations->get_len()-1) << endl;
    
    string data_file_name = folder_name + "/data-" + file_name + ".csv";
    string eps_file_name = folder_name + "/eps-" + file_name + ".csv";
    
    if (verbose)
    {
        cout << "*********************" << endl;
        cout << "Solving from a = " << a0 << " to a = " << a1 << endl;
        cout << "Data file printing to: " << data_file_name << endl;
    }
    
    auto start = high_resolution_clock::now();

    ofstream data_file(data_file_name);
    
    print_csv(data_file);
    print_eps_file(eps_file_name);

    
    for(int i = 0; i < a_separations->get_len(); i++)
    {
        if(!ODEOneRun(1000, 1, a_separations->get_value(i), "no_file", false, false))
        {
            cout << "ERROR: interval not completed; a_low = " << a0 << ", a_high = " << a1 << endl;
            cout << "Interval " << i << " from ";
            if (i==0)
                cout << a0;
            else
                cout << a_separations->get_value(i-1);
            cout << " to " << a_separations->get_value(i) << ". Stopped at x = " << x_value << " with dx = " << dx_value << endl;

            data_file.close();
            
            return false;
        }
        
        print_csv(data_file);
    }    
    data_file.close();
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    if (verbose)
    {
        cout << endl << "Time elapsed: "
         << duration.count()/1000. << " seconds" << endl;
        cout << "steps rejected / total steps = " << total_ODE_rejected_steps << " / " << total_ODE_steps << " (" << 100-(100 * total_ODE_rejected_steps) / (total_ODE_rejected_steps+total_ODE_steps) << "% efficiency)" << endl;

    }
    
    delete a_separations;
    
    
    return true;
}

double derivatives::calc_Neff()
{
    double nu_dens = y_values->get_neutrino_density(x_value);
    return nu_dens;// * pow(y_values->get_temp(), -4) * (4./7.) * pow(11./4., 4./3.) * 30. / _PI_ / _PI_;

}