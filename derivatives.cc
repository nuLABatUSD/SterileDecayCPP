#include "ODESolve.hh"
#include "arrays.hh"
#include "freqs_ntT.hh"
#include "derivatives.hh"
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
    delete y_values;
}

dummy_vars* derivatives::retrieve_separations(){
    return y_values->get_separations();
}

void derivatives::update(double new_a_start, double new_a_end){
    y_values->eps_shift(new_a_start, new_a_end);
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

int main(){
    /*
    * These four terms and E_low and E_high are constrained in order for our dummy_vars object to function as intended. In particular, we 
    * must obey the inequality scales_num >= floor[log( a_end / a_start ) / log(E_low * (num - 11) / E_high) + 2]
    */
    double a_start = 0.1;
    double a_end = 0.25;
    int num = 101;
    int scale_num = 3;
    
    double ms = 300;
    double theta = 1.22e-5;
    double time = 0;
    double temp = 1 / a_start;

    double E_low = min_low(ms);
    double E_high = (ms + 1) / 2;
    int file_idx = 0;

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

    // testing code
    /*integration* inter = new integration(eps, 17);
    double* results = new double[6];
    double* results2 = new double[6];
    inter->populate_Fvvbar(input, 0);
    cout << eps->get_value(16) << ", " << eps->get_value(17) << ", " << inter->interior_integral(input, 16, 0) << endl;
    dummy_vars** p3 = inter->get_p3();
    for(int i = 0; i < p3[16]->get_len(); i++){
       cout << i << ", " << eps->get_value(i) << ", " << p3[16]->get_weight(i) << ", " << p3[16]->get_value(i) << endl;
    }
    cout << inter->K3(4.28058, 4.02878, 8.30936) <<endl;
    inter->whole_integral(input, 0.1, 0, results);
    inter->whole_integral(input, 0.1, 1, results2);
    for(int i = 0; i < 6; i++){
        double one = results[i];
        double two = results2[i];
        cout << one + two << endl;
        if(one < 0){
            one *= -1;
        } 
        if(two < 0){
            two *= -1;
        }
        cout << eps->get_value(17) << ", " << one << ", " << two << ", " << one - two << ", " << 200 * (one - two) / (one + two) << "%" << endl;
    }
    delete inter;
    delete results;
    delete results2;*/
    
    sim->set_ics(a_start, input, 0.01 * a_start);
    dummy_vars* a_separations = sim->retrieve_separations();
    
    for(int i = 0; i < a_separations->get_len(); i++){
        double a = a_separations->get_value(i);
        cout << a << endl;
        string name = "ou" + to_string(file_idx) + ".csv";
        file_idx++; 
        sim->run(100, 2, a, name);
        if(i != a_separations->get_len() - 1){
            sim->shift_x();
        }
    }

    for(int j = 1; j < scale_num - 1; j++){
        cout << j << ", " << a_high << ", " << file_idx <<  endl;
        a_low = a_high;
        a_high = scales->get_value(j + 1);
        sim->update(a_low, a_high);
        a_separations = sim->retrieve_separations();
        for(int i = 0; i < a_separations->get_len(); i++){
            double a = a_separations->get_value(i);
            cout << a << endl;
            string name = "ou" + to_string(file_idx) + ".csv"; 
            file_idx++;
            sim->run(100, 2, a, name);
            if(i != a_separations->get_len() - 1){
                sim->shift_x();
            }
        }
    }

    delete a_separations;
    delete scales;
    delete input;
    delete sim;
    delete eps;
    delete freqs;

    return 0;
}