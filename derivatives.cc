#include "ODESolve.hh"
#include "arrays.hh"
#include "freqs_ntT.hh"
#include "derivatives.hh"
#include "constants.hh"
#include <string>
#include <iostream>
#include <fstream>

using std::cout;
using std::endl;
using std::to_string;
using std::string;
using std::ofstream;


derivatives::derivatives(int num, double start, double end, double ms, double theta, dummy_vars* freqs, double ns, double time, double temp):ODESolve(){
    num_bins = num;
    a_start = start;
    a_end = end;
    sterile_mass = ms;
    mixing_angle = theta;

    y_values = new freqs_ntT(num, start, end, ms, theta, freqs, ns, time, temp);
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

int main(){
    double a_start = 0.1;
    double a_end = 1;
    int num = int(a_end * 50 + 1);
    double ms = 300;
    double theta = 1.22e-5;
    double time = 0;
    double temp = 1 / a_start;

    linspace_and_gl* eps = new linspace_and_gl(0, (300 + 1) / (2 * 0.1), num, 0);
    dummy_vars* freqs = new dummy_vars(6 * num);
    for(int i = 0; i < num; i++){
        double E = eps->get_value(i);
        double f = exp(-E) / (exp(-E) + 1);
        freqs->set_value(i, f);
        freqs->set_value(i + num, f);
        freqs->set_value(i + 2 * num, f);
        freqs->set_value(i + 3 * num, f);
        freqs->set_value(i + 4 * num, f);
        freqs->set_value(i + 5 * num, f);
    }
    
    double ns =  (3 * _zeta_3_ / (2 * pow(_PI_,2))) * _gwd_ * pow(10,3) / _gsdec_;
    freqs_ntT* input = new freqs_ntT(num, a_start, a_end, ms, theta, freqs, ns, time, temp);
    freqs_ntT* output = new freqs_ntT(num, a_start, a_end, ms, theta, freqs, ns, time, temp);
    derivatives* sim = new derivatives(num, a_start, a_end, ms, theta, freqs, ns, time, temp);
    sim->set_ics(a_start, input, 0.01 * a_start);
    dummy_vars* a_separations = input->get_separations();
    cout << "Length:" << a_separations->get_len() << endl;
    for(int i = 0; i < a_separations->get_len(); i++){
        double a = a_separations->get_value(i);
        cout << a << endl;
        string name = "output" + to_string(i) + ".csv"; 
        sim->run(100, 2, a, name);
        sim->shift_x();
    }
    delete eps;
    delete freqs;
    delete input;
    delete sim;
    delete a_separations;
    return 0;
}