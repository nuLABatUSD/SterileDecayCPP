#include "ODESolve.hh"
#include "arrays.hh"
#include "freqs_ntT.hh"
#include "derivatives.hh"
#include "constants.hh"

using std::cout;
using std::endl;

derivatives::derivatives(double ms, double theta, double end){
    sterile_mass = ms;
    mixing_angle = theta;
    a_end = end;
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

int derivatives::get_bins(){
    return int (50 * a_end) + 1; 
}

void derivatives::f(double a, freqs_ntT* inputs, freqs_ntT* derivs){
    double num_bins = inputs->get_num_bins();
    dummy_vars* electron = new dummy_vars(num_bins);
    dummy_vars* anti_electron = new dummy_vars(num_bins);
    dummy_vars* muon = new dummy_vars(num_bins);
    dummy_vars* anti_muon = new dummy_vars(num_bins);
    dummy_vars* tau = new dummy_vars(num_bins);
    dummy_vars* anti_tau = new dummy_vars(num_bins);

    inputs->compute_dfda(a, electron, anti_electron, muon, anti_muon, tau, anti_tau);
    for(int i = 0; i < num_bins; i++){
        derivs->set_value(i, electron->get_value(i));
        derivs->set_value(i + num_bins, anti_electron->get_value(i));
        derivs->set_value(i + 2 * num_bins, muon->get_value(i));
        derivs->set_value(i + 3 * num_bins, anti_muon->get_value(i));
        derivs->set_value(i + 4 * num_bins, tau->get_value(i));
        derivs->set_value(i + 5 * num_bins, anti_tau->get_value(i));
    }

    double dnsda = inputs->get_dnsda(a);
    double dtda = inputs->get_dtda(a);
    double dTda = inputs->get_dTda(a);

    derivs->set_ns(dnsda);
    derivs->set_time(dtda);
    derivs->set_temp(dTda);

    delete electron;
    delete anti_electron;
    delete muon;
    delete anti_muon;
    delete tau;
    delete anti_tau;
}

int main(){
    int num = 100;
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
    freqs_ntT* input = new freqs_ntT(num, 0.1, 10, 300, 1.22e-5, eps, freqs, ns, 0, 10);
    input->get_separations()->print_all();
    freqs_ntT* derivs = new freqs_ntT(num, 300, 1.22e-5);
    derivatives* der = new derivatives(300, 1.22e-5, 10);
    der->f(0.1, input, derivs);
    cout << "Here: " << endl;
    cout << derivs->get_ns() << endl;
    cout << derivs->get_time() << endl;
    cout << derivs->get_temp() << endl;

    delete eps;
    delete freqs;
    delete input;
    delete derivs;
    delete der;
    return 0;
}