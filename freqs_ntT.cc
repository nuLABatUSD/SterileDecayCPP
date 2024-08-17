#include <cmath>
#include "constants.hh"
#include "arrays.hh"
#include "freqs_ntT.hh"
#include "thermodynamics.h"
#include "decays.h"
#include "gl_vals.hh"
#include "gel_vals.hh"
#include <iomanip>

using std::cout;
using std::endl;
// freqs_ntT makes calculation for derivatives
freqs_ntT::freqs_ntT(int num, double low, double high, double start, double end, double ms, double theta, dummy_vars* freqs, double ns, double time, double temp):dep_vars(6*num + 3)
{
    num_bins = num;
    E_low = low;
    E_high = high;
    a_start = start;
    a_end = end;
    eps = new gel_linspace_gl(E_low * a_start, E_high * a_end, num);
    sterile_mass = ms;
    mixing_angle = theta;

    for(int i = 0; i < num; i++){
        values[i] = freqs->get_value(i);
        values[i + num] = freqs->get_value(i + num);
        values[i + 2 * num] = freqs->get_value(i + 2 * num);
        values[i + 3 * num] = freqs->get_value(i + 3 * num);
        values[i + 4 * num] = freqs->get_value(i + 4 * num);
        values[i + 5 * num] = freqs->get_value(i + 5 * num);
    }

    values[6 * num] = ns;
    values[6 * num + 1] = time;
    values[6 * num + 2] = temp;
}

freqs_ntT::freqs_ntT(freqs_ntT* copy_me):dep_vars(6 * copy_me->get_num_bins() + 3){
    num_bins = copy_me->get_num_bins();
    E_low = copy_me->get_low();
    E_high = copy_me->get_high();
    a_start = copy_me->get_a_start();
    a_end = copy_me->get_a_end();
    sterile_mass = copy_me->get_ms();
    mixing_angle = copy_me->get_theta();
    eps = new gel_linspace_gl(E_low * a_start, E_high * a_end, num_bins);

    for(int i = 0; i < 6 * num_bins + 3; i++){
        values[i] = copy_me->get_value(i);
    }
}

freqs_ntT::~freqs_ntT()
{
    delete eps;
}

void freqs_ntT::eps_shift(double new_a_start, double new_a_end){
    this->set_a_start(new_a_start);
    this->set_a_end(new_a_end);
    gel_linspace_gl* new_eps = new gel_linspace_gl(E_low * new_a_start, E_high * new_a_end, num_bins);
    // known points are in eps, target point in new_eps
    double* freqs_ntT = new double[6 * num_bins + 3];
    for(int i = 0; i < 6 * num_bins; i++){
        freqs_ntT[i] = values[i];
    }
    for(int p = 0; p < 6; p++){
        for(int k = 0; k < num_bins; k++){
            if(new_eps->get_value(k) < eps->get_value(num_bins - 1)){
                double new_val = 0;
                double new_x_val = new_eps->get_value(k);
                int key_id = 0;
                while(eps->get_value(key_id) <= new_x_val){
                    key_id++;
                }
                int ids[4] = {p * num_bins + key_id - 2, p * num_bins + key_id - 1, p * num_bins + key_id, p * num_bins + key_id + 1};
                if(key_id + 1 >= num_bins){
                    ids[3] = p * num_bins + key_id - 3;
                }
                if(key_id - 2 < 0){
                    ids[0] = p * num_bins + key_id + 2;
                    if(key_id - 1 < 0){
                        ids[1] = p * num_bins + key_id + 3;
                    }
                }
                for(int i = 0; i < 4; i++){
                    
                    double multiplier = 1;
                    int old_idx = ids[i] % num_bins;
                    double old_x_val = eps->get_value(old_idx);
                    double old_val = freqs_ntT[p * num_bins + old_idx];
                    for(int j = 0; j < 4; j++){
                        int old_idx2 = ids[j] % num_bins;
                        double mult_x = eps->get_value(old_idx2);
                        if(i != j){
                            multiplier *= (new_x_val - mult_x) / (old_x_val - mult_x);
                        }
                    }
                    new_val += multiplier * log10(old_val);
                }
                values[p * num_bins + k] = pow(10, new_val);
            } else {
                double old_eps1 = eps->get_value(num_bins - 2);
                double old_f1 = freqs_ntT[(p + 1) * num_bins - 2];
                double old_eps2 = eps->get_value(num_bins - 1);
                double old_f2 = freqs_ntT[(p + 1) * num_bins - 1];

                double logy = ((new_eps->get_value(k) - old_eps1) * (log(old_f2) - log(old_f1)) / (old_eps2 - old_eps1)) + log(old_f1);
                values[p * num_bins + k] = exp(logy);
            }
        }
    }
    eps = new gel_linspace_gl(E_low * new_a_start, E_high * a_end, num_bins);
    delete new_eps;
    delete[] freqs_ntT;
}

double freqs_ntT::get_eps_value(int index){
    return eps->get_value(index);
}

gel_linspace_gl* freqs_ntT::get_eps(){
    return eps;
}

double freqs_ntT::get_ms(){
    return sterile_mass;
}

double freqs_ntT::get_theta(){
    return mixing_angle;
}

int freqs_ntT::get_num_bins(){
    return num_bins;
}

double freqs_ntT::get_low(){
    return E_low;
}

double freqs_ntT::get_high(){
    return E_high;
}

double freqs_ntT::get_a_start(){
    return a_start;
}

double freqs_ntT::get_a_end(){
    return a_end;
}

void freqs_ntT::set_low(double new_low){
    E_low = new_low;
}

void freqs_ntT::set_high(double new_high){
    E_high = new_high;
}

void freqs_ntT::set_a_start(double start){
    a_start = start;
}

void freqs_ntT::set_a_end(double end){
    a_end = end;
}

void freqs_ntT::set_ms(double ms){
    sterile_mass = ms;
}

void freqs_ntT::set_theta(double theta){
    mixing_angle = theta;
}

double freqs_ntT::get_temp(){
    return values[6 * num_bins + 2];
}

void freqs_ntT::set_temp(double new_temp){
    values[6 * num_bins + 2] = new_temp;
}

double freqs_ntT::get_time(){
    return values[6 * num_bins + 1];
}

void freqs_ntT::set_time(double new_time){
    values[6 * num_bins + 1] = new_time;
}

double freqs_ntT::get_ns(){
    return values[6 * num_bins];
}

void freqs_ntT::set_ns(double new_ns){
    values[6 * num_bins] = new_ns;
}

double freqs_ntT::get_sterile_density(){
    return this->get_ms() * this->get_ns();
}

double freqs_ntT::get_photon_density(){
    return pow(_PI_, 2) * pow(this->get_temp(), 4) / 15;
}

double freqs_ntT::get_ind_neutrino_density(int identifier, double a){
    dep_vars* integrand = new dep_vars(num_bins);
    double multiplier = 1 / (2 * pow(a, 4) * pow(_PI_, 2));
    for(int i = 0; i < num_bins; i++){
        integrand->set_value(i, multiplier * pow(eps->get_value(i), 3) * values[i + identifier * num_bins]);
    }
    //eps->set_trap_weights(); Modified the integration
    double integral = eps->integrate(integrand);

    delete integrand;

    return integral;
}

double freqs_ntT::get_neutrino_density(double a){
    double density = 0;
    for(int i = 0; i < 6; i++){
        density += get_ind_neutrino_density(i, a);
    }
    return density;
}

double freqs_ntT::get_dtda(double a){
    double sterile_density = this->get_sterile_density();
    double photon_density = this->get_photon_density();
    double neutrino_density = this->get_neutrino_density(a);
    double* pe_density = new double;
    double* pe_pressure = new double;
    energy_and_pressure(_electron_mass_, this->get_temp(), pe_density, pe_pressure);
    double total_density = sterile_density + photon_density + neutrino_density + *pe_density;

    delete pe_density;
    delete pe_pressure;

    return sqrt(3 * pow(_planck_mass_, 2) / (8 * _PI_)) / (a * sqrt(total_density));
}

double freqs_ntT::get_dnsda(double a){
    double lifetime = get_lifetime(sterile_mass, mixing_angle);
    double ns = this->get_ns();
    double dtda = this->get_dtda(a);
    return - (1 / lifetime) * ns * dtda - 3 * ns / a;
}

void freqs_ntT::compute_dfda(double a, dummy_vars* electron, dummy_vars* anti_electron, dummy_vars* muon, dummy_vars* anti_muon, dummy_vars* tau, dummy_vars* anti_tau){
    dummy_vars* e = new dummy_vars(num_bins);
    dummy_vars* ae = new dummy_vars(num_bins);
    dummy_vars* m = new dummy_vars(num_bins);
    dummy_vars* am = new dummy_vars(num_bins);
    dummy_vars* t = new dummy_vars(num_bins);
    dummy_vars* at = new dummy_vars(num_bins);
    
    compute_full_term(eps, sterile_mass, mixing_angle, 1 / a, e, ae, m, am, t, at);
    double dtda = this->get_dtda(a);
    double ns = this->get_ns();
    for(int i = 0; i < num_bins; i++){
        electron->set_value(i, e->get_value(i) * ns * dtda);
        anti_electron->set_value(i, ae->get_value(i) * ns * dtda);
        muon->set_value(i, m->get_value(i) * ns * dtda);
        anti_muon->set_value(i, am->get_value(i) * ns * dtda);
        tau->set_value(i, t->get_value(i) * ns * dtda);
        anti_tau->set_value(i, at->get_value(i) * ns * dtda);
    }

    delete e;
    delete ae;
    delete m;
    delete am;
    delete t;
    delete at;
}

void freqs_ntT::compute_dfda_cube(double a, dep_vars* dfda_cube){
    dummy_vars* e = new dummy_vars(num_bins);
    dummy_vars* ae = new dummy_vars(num_bins);
    dummy_vars* m = new dummy_vars(num_bins);
    dummy_vars* am = new dummy_vars(num_bins);
    dummy_vars* t = new dummy_vars(num_bins);
    dummy_vars* at = new dummy_vars(num_bins);

    compute_dfda(a, e, ae, m, am, t, at);
    for(int i = 0; i < num_bins; i++){
        double sum = e->get_value(i) + ae->get_value(i) + m->get_value(i) + am->get_value(i) + t->get_value(i) + at->get_value(i);
        dfda_cube->set_value(i, sum * pow(eps->get_value(i), 3));
    }

    delete e;
    delete ae;
    delete m;
    delete am;
    delete t;
    delete at;
}

double freqs_ntT::get_dQda(double a){
    double lifetime = get_lifetime(sterile_mass, mixing_angle);
    double dtda = get_dtda(a);
    double ns = get_ns();

    double total_flow = sterile_mass * ns * pow(a, 3) * dtda / lifetime;
    //eps->set_trap_weights(); Modified the integration

    dep_vars* dfda_eps_cube = new dep_vars(num_bins);
    compute_dfda_cube(a, dfda_eps_cube);
    double neutrino_loss = eps->integrate(dfda_eps_cube) / (2  * pow(_PI_, 2) * a);

    delete dfda_eps_cube;
    return total_flow - neutrino_loss;
}

double freqs_ntT::get_dTda(double a){
    double temp = this->get_temp();
    double* pe_density = new double;
    double* pe_pressure = new double;
    double* pe_density_temp = new double;
    double* pe_pressure_temp = new double;

    energy_pressure_and_derivs(_electron_mass_, temp, pe_density, pe_pressure, pe_density_temp, pe_pressure_temp);

    double dQda = this->get_dQda(a);
    double term_0 = (1 / temp) * dQda;
    double term_1 = 4 * pow(_PI_, 2) * pow(temp, 3) / 45;
    double term_2 = (*pe_density + *pe_pressure) / temp;
    double term_3 = (3 / temp) * term_1;
    double term_4 = -(1 / pow(temp, 2)) * (*pe_density + *pe_pressure);
    double term_5 = (1 / temp) * (*pe_density_temp + *pe_pressure_temp);

    delete pe_density;
    delete pe_pressure;
    delete pe_density_temp;
    delete pe_pressure_temp;

    return (term_0 - 3 * pow(a, 2) * (term_1 + term_2)) / (pow(a, 3) * (term_3 + term_4 + term_5));
}

void freqs_ntT::compute_derivs(double a, dummy_vars* electron, dummy_vars* anti_electron, dummy_vars* muon, dummy_vars* anti_muon, dummy_vars* tau, dummy_vars* anti_tau, double* dnsda, double* dtda, double* dTda)
{
    // dtda calculations
    double sterile_density = this->get_sterile_density();
    double photon_density = this->get_photon_density();
    double neutrino_density = this->get_neutrino_density(a);
    double temp = this->get_temp();
    double* pe_density = new double;
    double* pe_pressure = new double;
    double* pe_density_temp = new double;
    double* pe_pressure_temp = new double;

    energy_pressure_and_derivs(_electron_mass_, temp, pe_density, pe_pressure, pe_density_temp, pe_pressure_temp);
    double total_density = sterile_density + photon_density + neutrino_density + *pe_density;

    *dtda = sqrt(3 * pow(_planck_mass_, 2) / (8 * _PI_)) / (a * sqrt(total_density));

    // dnsda calculations
    double lifetime = get_lifetime(sterile_mass, mixing_angle);
    double ns = this->get_ns();
    *dnsda = - (1 / lifetime) * ns * *dtda - 3 * ns / a;

    // dfda calculations
    dummy_vars* e = new dummy_vars(num_bins);
    dummy_vars* ae = new dummy_vars(num_bins);
    dummy_vars* m = new dummy_vars(num_bins);
    dummy_vars* am = new dummy_vars(num_bins);
    dummy_vars* t = new dummy_vars(num_bins);
    dummy_vars* at = new dummy_vars(num_bins);
    
    compute_full_term(eps, sterile_mass, mixing_angle, 1 / a, e, ae, m, am, t, at);
    for(int i = 0; i < num_bins; i++){
        electron->set_value(i, e->get_value(i) * ns * *dtda);
        anti_electron->set_value(i, ae->get_value(i) * ns * *dtda);
        muon->set_value(i, m->get_value(i) * ns * *dtda);
        anti_muon->set_value(i, am->get_value(i) * ns * *dtda);
        tau->set_value(i, t->get_value(i) * ns * *dtda);
        anti_tau->set_value(i, at->get_value(i) * ns * *dtda);
    }
    
    delete e;
    delete ae;
    delete m;
    delete am;
    delete t;
    delete at;

    dep_vars* dfda_eps_cube = new dep_vars(num_bins);
    for(int i = 0; i < num_bins; i++){
        double sum = electron->get_value(i) + anti_electron->get_value(i) + muon->get_value(i) + anti_muon->get_value(i) + tau->get_value(i) + anti_tau->get_value(i);
        dfda_eps_cube->set_value(i, sum * pow(eps->get_value(i), 3));
    }

    double total_flow = sterile_mass * ns * pow(a, 3) * *dtda / lifetime;
    //eps->set_trap_weights(); Modified the integration
    double neutrino_loss = eps->integrate(dfda_eps_cube) / (2  * pow(_PI_, 2) * a);

    delete dfda_eps_cube;
    double dQda = total_flow - neutrino_loss;

    double term_0 = (1 / temp) * dQda;
    double term_1 = 4 * pow(_PI_, 2) * pow(temp, 3) / 45;
    double term_2 = (*pe_density + *pe_pressure) / temp;
    double term_3 = (3 / temp) * term_1;
    double term_4 = -(1 / pow(temp, 2)) * (*pe_density + *pe_pressure);
    double term_5 = (1 / temp) * (*pe_density_temp + *pe_pressure_temp);

    delete pe_density;
    delete pe_pressure;
    delete pe_density_temp;
    delete pe_pressure_temp;

    *dTda = (term_0 - 3 * pow(a, 2) * (term_1 + term_2)) / (pow(a, 3) * (term_3 + term_4 + term_5));
}

dummy_vars* freqs_ntT::get_separations(){
    // max size listing of separations
    dummy_vars* overfull = new dummy_vars(10 * num_bins);
    double index = 0;
    // initializing pointers for kinematic quantities
    double* waste = new double;
    double* gamma_muon_3 = new double;
    double* muon_speed_3 = new double;
    double* gamma_pion_mu = new double;
    double* pion_speed_mu = new double;
    double* gamma_pion_e = new double;
    double* pion_speed_e = new double;
    double* gamma_muon_4 = new double;
    double* muon_speed_4 = new double;

    // kinetic terms
    compute_kinetics(sterile_mass, _muon_mass_, _charged_pion_mass_, gamma_muon_3, muon_speed_3, waste);
    compute_kinetics(sterile_mass, _charged_pion_mass_, _muon_mass_, gamma_pion_mu, pion_speed_mu, waste);
    compute_kinetics(sterile_mass, _charged_pion_mass_, _electron_mass_, gamma_pion_e, pion_speed_e, waste);
    compute_kinetics(_charged_pion_mass_, _muon_mass_, 0, gamma_muon_4, muon_speed_4, waste);

    // type I accounting
    double nenergy1 = get_monoenergy(sterile_mass, 0, 0);
    double nenergy2 = get_monoenergy(sterile_mass, 0, _neutral_pion_mass_);

    // type II accounting
    double neutrino_energy_pion = get_monoenergy(_charged_pion_mass_, 0, _muon_mass_);
    double min_energy_mu = *gamma_pion_mu * neutrino_energy_pion * (1 - *pion_speed_mu);
    double max_energy_mu = *gamma_pion_mu * neutrino_energy_pion * (1 + *pion_speed_mu);
    double min_energy_e = *gamma_pion_e * neutrino_energy_pion * (1 - *pion_speed_e);
    double max_energy_e = *gamma_pion_e * neutrino_energy_pion * (1 + *pion_speed_e);

    // type III accounting
    double neutrino_max_energy_muon = get_monoenergy(_muon_mass_, 0, _electron_mass_);
    double nenergy3 = *gamma_muon_3 * (1 + *muon_speed_3) * neutrino_max_energy_muon;
    double nenergy32 = *gamma_muon_3 * (1 - *muon_speed_3) * neutrino_max_energy_muon;

    // type IV accounting
    double nenergy4 = neutrino_max_energy_muon * (*gamma_pion_mu * (1 + *pion_speed_mu) * *gamma_muon_4 * (1 + *muon_speed_4));
    double nenergy42 = neutrino_max_energy_muon * (*gamma_pion_e * (1 + *pion_speed_e) * *gamma_muon_4 * (1 + *muon_speed_4));

    // assignment to overfull listing
    for(int i = 0; i < num_bins; i++){
        double energy = eps->get_value(i);
        overfull->set_value(index, energy / nenergy1);
        index += 1;
        if(sterile_mass > _neutral_pion_mass_){
            overfull->set_value(index, energy / nenergy2);
            index += 1;
            if(sterile_mass > _charged_pion_mass_ + _electron_mass_){
                overfull->set_value(index, energy / min_energy_e);
                index += 1;
                overfull->set_value(index, energy / max_energy_e);
                index += 1;
                overfull->set_value(index, energy / nenergy42);
                index += 1;
                if(sterile_mass > _charged_pion_mass_ + _muon_mass_){
                    overfull->set_value(index, energy / nenergy3);
                    index += 1;
                    overfull->set_value(index, energy / nenergy32);
                    index += 1;
                    overfull->set_value(index, energy / nenergy4);
                    index += 1;
                    overfull->set_value(index, energy / min_energy_mu);
                    index += 1;
                    overfull->set_value(index, energy / max_energy_mu);
                    index += 1;
                }
            }
        }
    }
    
    delete waste; 
    delete gamma_muon_3; 
    delete muon_speed_3; 
    delete gamma_pion_mu; 
    delete pion_speed_mu;
    delete gamma_pion_e; 
    delete pion_speed_e; 
    delete gamma_muon_4; 
    delete muon_speed_4;

    // conditioning endpoints
    double num_switches = 0;
    for(int i = 0; i < index; i++){
        double val = overfull->get_value(i);
        if(val <= a_start || val >= a_end){
            overfull->set_value(i, 0);
            num_switches += 1;
        }
    }
    // sort listings
    bool swapped;
    for(int i = 0; i < index - 1; i++){
        swapped = false;
        for(int j = 0; j < index - i - 1; j++){
            if(overfull->get_value(j) > overfull->get_value(j+1)){
                double swap = overfull->get_value(j+1);
                overfull->set_value(j+1, overfull->get_value(j));
                overfull->set_value(j, swap);
                swapped = true;
            }
        }
        if(swapped == false){
            break;
        }
    }
    
    dummy_vars* reduced = new dummy_vars(index - num_switches + 1);
    reduced->set_value(index - num_switches, a_end);
    for(int i = 0; i < index - num_switches; i++){
        reduced->set_value(i, overfull->get_value(i + 10 * num_bins - index + num_switches));
    }
    delete overfull;
    return reduced;
}

// integration used to make computations for collision integrals

integration::integration(gel_linspace_gl* e, int p1_idx){
    eps = new gel_linspace_gl(e);
    p1 = p1_idx;
    count = 0;

    int size = eps->get_len();

    outer_vals = new dep_vars(size);
    inner_vals = new dep_vars*[size];
    p3_vals = new dummy_vars*[size];

    for(int p2 = 0; p2 < size; p2++){
        double p1p2 = eps->get_value(p2) + eps->get_value(p1);
        int min = eps->get_min_linspace();
        int max = eps->get_max_linspace();
        // special case can be added for efficiency (needs modification)
        /*if(p1 >= min && p1 <= max && p2 >= min && p2 <= max && p1p2 >= min && p1p2 <= max){
            //int bottom_lin = eps->get_min_linspace() / eps->get_delta();
            p3_vals[p2] = new dummy_vars(p1 + p2 - 1);
            int i = 0;
            for(int i = 0; i < p1 + p2 - 1; i++){
                p3_vals[p2]->set_value(i, eps->get_value(i));
            }
            p3_vals[p2]->set_trap_weights();
            inner_vals[p2] = new dep_vars(p1 + p2 - 1);
            cout << "Index: " << p2 << ", p1 + p2 = " << eps->get_value(p1)+eps->get_value(p2) << ", p3 =" << p3_vals[p2]->get_value(p3_vals[p2]->get_len() - 1)<<endl; // remove
        }*/
        //else{
        //count will give the number of energy values in eps that are less than or equal to the energy of p1+p2
        //this meants that count-1 will give the index of greatest element of eps less than the energy of p1+p2
        //furthermore, if count is bigger than N, the energy of p1+p2 is bigger than the biggest element of eps
        count = 0;
        for(int i = 0; i < size; i++){
            if(eps->get_value(i) < p1p2){
                count++;
            }
        }

        //p3_vals[p2] will have count+1 elements because we want count elements from eps as well as p1+p2
        p3_vals[p2] = new dummy_vars(count + 1);
        for(int i = 0; i < count; i++){
            p3_vals[p2]->set_value(i, eps->get_value(i));
        }
        p3_vals[p2]->set_value(count, p1p2);
        p3_vals[p2]->set_trap_weights();
        p3_vals[p2]->set_weight(0, p3_vals[p2]->get_weight(0) + p3_vals[p2]->get_value(0) / 2);
        inner_vals[p2] = new dep_vars(count + 1);
        //}
    }

    Fvv_values = new double**[6]; 
    Fvvbar_values = new double**[6];
    for(int i=0; i<6; i++){
        Fvv_values[i] = new double*[eps->get_len()];
        Fvvbar_values[i] = new double*[eps->get_len()];
        for(int j=0; j<eps->get_len(); j++){
            Fvv_values[i][j] = new double[eps->get_len() + 1](); 
            Fvvbar_values[i][j] = new double[eps->get_len() + 1]();  
        }
    }    
}

integration::~integration(){
    delete outer_vals;
    for(int i=0; i<eps->get_len(); i++){
        delete inner_vals[i];
        delete p3_vals[i];
    }
    delete[] inner_vals;
    delete[] p3_vals;
    for(int i=0; i<6; i++){
        for(int j=0; j<eps->get_len(); j++){
            delete[] Fvv_values[i][j];
            delete[] Fvvbar_values[i][j];
        }
        delete[] Fvv_values[i];
        delete[] Fvvbar_values[i];
    }
    delete[] Fvv_values;
    delete[] Fvvbar_values;
    delete eps;

}

dummy_vars** integration::get_p3(){
    return p3_vals;
}

double integration::Fvv_comp(freqs_ntT* input, bool neutrino, int which_term, int p2, int p3, double check){
    double F = 0;
    double p1_energy = eps->get_value(p1);
    double p2_energy = eps->get_value(p2);
    double p3_energy = p3_vals[p2]->get_value(p3);
    double p4_energy = p1_energy + p2_energy - p3_energy;

    int num = input->get_num_bins();
    int k_mod = 0;

    double f1 = input->get_value(which_term * num + p1);
    double f2 = 0;
    double f3 = 0;
    double f4 = 0;
    for(int k = 0; k < 3; k++){
        if(neutrino){
            k_mod = 2 * k;
        } else {
            k_mod = 2 * k + 1;
        }
        f4 = 0;
        double s_mod = 1.;
        if(k_mod != which_term){
            s_mod = 1./2;
        }
        // f2 and f3 (if p3 not end-point) known exactly
        f2 = input->get_value(k_mod * num + p2);
        if(p3 < p3_vals[p2]->get_len() - 1){
            f3 = input->get_value(which_term * num + p3);
        } else {
            if(p3_energy <= eps->get_value(num)){
                double new_val = 0;
                int ids[4] = {which_term * num + count - 2, which_term * num + count - 1, which_term * num + count, which_term * num + count + 1};
                if(p3 + 1 >= num){
                    ids[3] = which_term * num + count - 3;
                }
                if(p3 - 2 < 0){
                    ids[0] = which_term * num + count + 2;
                    if(p3 - 1 < 0){
                        ids[1] = which_term * num + count + 3;
                    }
                }
                for(int i = 0; i < 4; i++){
                    
                    double multiplier = 1;
                    int old_idx = ids[i] % num;
                    double old_x_val = eps->get_value(old_idx);
                    double old_val = input->get_value(which_term * num + old_idx);
                    for(int j = 0; j < 4; j++){
                        int old_idx2 = ids[j] % num;
                        double mult_x = eps->get_value(old_idx2);
                        if(i != j){
                            multiplier *= (p3_energy - mult_x) / (old_x_val - mult_x);
                        }
                    }
                    new_val += multiplier * log10(old_val);
                }
                f3 = pow(10, new_val);
            } else {
                double old_eps1 = eps->get_value(num - 2);
                double old_f1 = input->get_value((which_term + 1) * num - 2);
                double old_eps2 = eps->get_value(num - 1);
                double old_f2 = input->get_value((which_term + 1) * num - 1);

                double logy = ((p3_energy - old_eps1) * (log(old_f2) - log(old_f1)) / (old_eps2 - old_eps1)) + log(old_f1);
                f3 = exp(logy);
            }
        }


        // interpolation / extrapolation for p4
        f4 = 0;
        if(eps->get_value(num - 1) >= p4_energy){
            int p4 = 0;
            for(int j = 0; j < num; j++){
                if(eps->get_value(j) <= p4_energy){
                    p4++;
                }
            }

            int ids[4] = {k_mod* num + p4 - 2, k_mod* num + p4 - 1, k_mod* num + p4, k_mod* num + p4 + 1};
            if(p4 + 1 >= num){
                ids[3] = k_mod* num + p4 - 3;
            }
            if(p4 - 2 < 0){
                ids[0] = k_mod* num + p4 + 2;
                if(p4 - 1 < 0){
                    ids[1] = k_mod* num + p4 + 3;
                }
            }
            
            for(int i = 0; i < 4; i++){
                double multiplier = 1;
                int old_idx = ids[i] % num;
                double old_x_val = eps->get_value(old_idx);
                double old_val = input->get_value(ids[i]);
                for(int j = 0; j < 4; j++){
                    int old_idx2 = ids[j] % num;
                    double mult_x = eps->get_value(old_idx2);
                    if(i != j){
                        multiplier *= (p4_energy - mult_x) / (old_x_val - mult_x);
                    }
                }
                f4 += multiplier * log10(old_val);
            }
            f4 = pow(10, f4);  
        } else {
            double old_eps1 = eps->get_value(num - 2);
            double old_f1 = input->get_value((k_mod+ 1) * num - 2);
            double old_eps2 = eps->get_value(num - 1);
            double old_f2 = input->get_value((k_mod+ 1) * num - 1);

            double logy = ((p4_energy - old_eps1) * (log(old_f2) - log(old_f1)) / (old_eps2 - old_eps1)) + log(old_f1);
            f4 = exp(logy);
        }
        F += s_mod * (abs(check - 1) * f3 * f4 * (1 - f1) * (1 - f2) - check * f1 * f2 * (1 - f3) * (1 - f4));
    }
    return F;
}

void integration::populate_Fvv(freqs_ntT* input, double check){
    int num = eps->get_len();
    for(int i = 0; i < num; i++){
        int num3 = p3_vals[i]->get_len();
        for(int j = 0; j < num + 1; j++){
            for(int k = 0; k < 6; k++){
                bool neutrino = false;
                if(k % 2 == 0){
                    neutrino = true;
                }
                if(j < num3){
                    Fvv_values[k][i][j] = Fvv_comp(input, neutrino, k, i, j, check);
                } else {
                    Fvv_values[k][i][j] = 0;
                }
            }
        }
    }
}

double integration::Fvvbar_comp(freqs_ntT* input, bool neutrino, int which_term, int p2, int p3, double check){
    double F = 0;
    double p1_energy = eps->get_value(p1);
    double p2_energy = eps->get_value(p2);
    double p3_energy = p3_vals[p2]->get_value(p3);
    double p4_energy = p1_energy + p2_energy - p3_energy;

    int num = input->get_num_bins();

    double f1 = 0;
    double f2 = 0;
    double f3 = 0;
    double f4 = 0;

    int v1;
    int v2;
    int v3;
    int v4;
    int term = which_term / 2;
    if(neutrino){
        v1 = which_term;
        f1 = input->get_value(which_term * num + p1);
    } else {
        v2 = which_term;
        f2 = input->get_value(which_term * num + p2);
    }
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            double f3 = 0;
            double f4 = 0;
            if(i == term || i == j){
                double s_mod = 1;
                if(neutrino){
                    v2 = 2 * i + 1;
                    f2 = input->get_value(v2 * num + p2);
                } else {
                    v1 = 2 * i;
                    f1 = input->get_value(v1 * num + p1);
                }

                if(v2 == v1 + 1){
                    v3 = 2 * j + 1;
                    v4 = 2 * j;
                    if(v1 != v4){
                        s_mod = 1./4;
                    }
                } else {
                    v3 = v2;
                    v4 = v1;
                    s_mod = 1./4;
                }

                //determine f3, f4
                if(p3 < p3_vals[p2]->get_len() - 1){
                    f3 = input->get_value(v3 * num + p3);
                } else {
                    if(p3_energy <= eps->get_value(num)){
                        double new_val = 0;
                        int ids[4] = {v3 * num + count - 2, v3 * num + count - 1, v3 * num + count, v3 * num + count + 1};
                        if(p3 + 1 >= num){
                            ids[3] = v3 * num + count - 3;
                        }
                        if(p3 - 2 < 0){
                            ids[0] = v3 * num + count + 2;
                            if(p3 - 1 < 0){
                                ids[1] = v3 * num + count + 3;
                            }
                        }
                        for(int i = 0; i < 4; i++){
                            
                            double multiplier = 1;
                            int old_idx = ids[i] % num;
                            double old_x_val = eps->get_value(old_idx);
                            double old_val = input->get_value(v3 * num + old_idx);
                            for(int j = 0; j < 4; j++){
                                int old_idx2 = ids[j] % num;
                                double mult_x = eps->get_value(old_idx2);
                                if(i != j){
                                    multiplier *= (p3_energy - mult_x) / (old_x_val - mult_x);
                                }
                            }
                            new_val += multiplier * log10(old_val);
                        }
                        f3 = pow(10, new_val);
                    } else {
                        double old_eps1 = eps->get_value(num - 2);
                        double old_f1 = input->get_value((v3 + 1) * num - 2);
                        double old_eps2 = eps->get_value(num - 1);
                        double old_f2 = input->get_value((v3 + 1) * num - 1);

                        double logy = ((p3_energy - old_eps1) * (log(old_f2) - log(old_f1)) / (old_eps2 - old_eps1)) + log(old_f1);
                        f3 = exp(logy);
                    }
                }


                // interpolation / extrapolation for p4
                if(eps->get_value(num - 1) >= p4_energy){
                    int p4 = 0;
                    for(int j = 0; j < num; j++){
                        if(eps->get_value(j) < p4_energy){
                            p4++;
                        }
                    }

                    int ids[4] = {v4 * num + p4 - 2, v4 * num + p4 - 1, v4 * num + p4, v4 * num + p4 + 1};
                    
                    if(p4 + 1 >= num){
                        ids[3] = v4 * num + p4 - 3;
                    }
                    if(p4 - 2 < 0){
                        ids[0] = v4 * num + p4 + 2;
                        if(p4 - 1 < 0){
                            ids[1] = v4 * num + p4 + 3;
                        }
                    }
                    
                    
                    for(int i = 0; i < 4; i++){
                        double multiplier = 1;
                        int old_idx = ids[i] % num;
                        double old_x_val = eps->get_value(old_idx);
                        double old_val = input->get_value(ids[i]);
                        for(int j = 0; j < 4; j++){
                            int old_idx2 = ids[j] % num;
                            double mult_x = eps->get_value(old_idx2);
                            if(i != j){
                                multiplier *= (p4_energy - mult_x) / (old_x_val - mult_x);
                            }
                        }
                        f4 += multiplier * log10(old_val);
                    }
                    f4 = pow(10, f4); 
                } else {
                    double old_eps1 = eps->get_value(num - 2);
                    double old_f1 = input->get_value((v4 + 1) * num - 2);
                    double old_eps2 = eps->get_value(num - 1);
                    double old_f2 = input->get_value((v4 + 1) * num - 1);

                    double logy = ((p4_energy - old_eps1) * (log(old_f2) - log(old_f1)) / (old_eps2 - old_eps1)) + log(old_f1);
                    f4 = exp(logy);
                }
                F += s_mod * (abs(check - 1) * f3 * f4 * (1 - f1) * (1 - f2) - check * f1 * f2 * (1 - f3) * (1 - f4));
            }
        }
    }
    return F;
}

void integration::populate_Fvvbar(freqs_ntT* input, double check){
    int num = eps->get_len();
    for(int i = 0; i < num; i++){
        int num3 = p3_vals[i]->get_len();
        for(int j = 0; j < num + 1; j++){
            for(int k = 0; k < 6; k++){
                bool neutrino = false;
                if(k % 2 == 0){
                    neutrino = true;
                }
                if(j < num3){
                    Fvvbar_values[k][i][j] = Fvvbar_comp(input, neutrino, k, i, j, check) / 4;
                } else {
                    Fvvbar_values[k][i][j] = 0;
                }
            }
        }
    }
}

double integration::J1(double p1, double p2, double p3){
    return (16. / 15) * pow(p3, 3) * (10 * pow(p1 + p2, 2) - 15 * (p1 + p2) * p3 + 6 * pow(p3, 2));
}

double integration::J2(double p1, double p2){
    return (16. / 15) * pow(p2, 3) * (10 * pow(p1, 2) + 5 * p1 * p2 + pow(p2, 2));
}

double integration::J3(double p1, double p2, double p3){
    return (16. / 15) * (pow(p1 + p2, 5) - 10 * pow(p1 + p2, 2) * pow(p3, 3) + 15 * (p1 + p2) * pow(p3, 4) - 6 * pow(p3, 5));
}

double integration::K1(double p1, double p3){
    return 16./15 * pow(p3,3) * (10 * pow(p1,2) - 5 * p1*p3 + pow(p3,2));
}

double integration::K2(double p1, double p2, double p3){
    return 16./15 * pow(p2,3) * (10 * pow(p1-p3,2) + 15 * (p1-p3) * p2 + 6 * pow(p2,2));
}

double integration::K3(double p1, double p2, double p3){
    if(p1 + p2 == p3){
        return 0;
    }
    return 16./15 * (pow(p1-p3,5) + 10 * pow(p1-p3,2) * pow(p2,3) + 15 * (p1-p3) * pow(p2,4) + 6 * pow(p2,5));
}

double integration::interior_integral(int p2, int which_term){
    // split finite and infinite terms
    double p_1_energy = eps->get_value(p1);
    double p2_energy = eps->get_value(p2);
    if(p2<p1){
        int p3 = 0;
        while(p3_vals[p2]->get_value(p3) < p2_energy){
            inner_vals[p2]->set_value(p3, Fvv_values[which_term][p2][p3] * J1(p_1_energy, p2_energy, p3_vals[p2]->get_value(p3)) + Fvvbar_values[which_term][p2][p3] * K1(p_1_energy, p3_vals[p2]->get_value(p3)));
            p3++;
        }
        while(p3_vals[p2]->get_value(p3) < p_1_energy){
            inner_vals[p2]->set_value(p3, Fvv_values[which_term][p2][p3] * J2(p_1_energy, p2_energy) + Fvvbar_values[which_term][p2][p3] * K2(p_1_energy, p2_energy, p3_vals[p2]->get_value(p3)));
            p3++;
        }
        while(p3 < inner_vals[p2]->length()){
            inner_vals[p2]->set_value(p3, Fvv_values[which_term][p2][p3] * J3(p_1_energy, p2_energy, p3_vals[p2]->get_value(p3)) + Fvvbar_values[which_term][p2][p3] * K3(p_1_energy, p2_energy, p3_vals[p2]->get_value(p3)));
            p3++;
        }
        
    }
    
    else{
        int p3 = 0;
        while(p3_vals[p2]->get_value(p3) < p_1_energy){
            inner_vals[p2]->set_value(p3, Fvv_values[which_term][p2][p3] * J1(p_1_energy, p2_energy, p3_vals[p2]->get_value(p3)) + Fvvbar_values[which_term][p2][p3] * K1(p_1_energy, p3_vals[p2]->get_value(p3)));
            p3++;
        }
        while(p3_vals[p2]->get_value(p3) < p2_energy){
            inner_vals[p2]->set_value(p3, Fvv_values[which_term][p2][p3] * J2(p2_energy, p_1_energy) + Fvvbar_values[which_term][p2][p3] * K1(p3_vals[p2]->get_value(p3), p_1_energy));
            p3++;
        }
        while(p3 < inner_vals[p2]->length()){
            inner_vals[p2]->set_value(p3, Fvv_values[which_term][p2][p3] * J3(p_1_energy, p2_energy, p3_vals[p2]->get_value(p3)) + Fvvbar_values[which_term][p2][p3] * K3(p_1_energy, p2_energy, p3_vals[p2]->get_value(p3)));
            p3++;
        }
    }
    
    double result = p3_vals[p2]->integrate(inner_vals[p2]);
    return result;
}

void integration::whole_integral(freqs_ntT* input, double a, double check, double* results){
    double p_1_energy = eps->get_value(p1);
    if (p_1_energy == 0){
        for(int i=0; i<6; i++){
            results[i] = 0;
        }
    }
    else{
        //populates F_values
        this->populate_Fvv(input, check);
        this->populate_Fvvbar(input, check);
        double Tcm = 1 / a;
        
        for(int i=0; i<6; i++){
            for(int p2=0; p2<eps->get_len(); p2++){
                outer_vals->set_value(p2, interior_integral(p2, i));
            }
            results[i] = eps->integrate(outer_vals);
            results[i] *= pow(Tcm, 5) * pow(_GF_,2) / (pow(2*_PI_,3) * pow(p_1_energy,2));
        }
    }
}

// nu_e_collision_R1 to take care of neutrino-electron R1 collision integrals

nu_e_collision_R1::nu_e_collision_R1(gel_linspace_gl* e, int p1_idx, double a){
    eps = new gel_linspace_gl(e);
    p1 = p1_idx;
    temp_cm = 1 / a;
    me_scaled = _electron_mass_ / temp_cm;
    count = 0;

    q2_vals = new dummy_vars(50);
    int size = q2_vals->get_len();

    for(int i = 0; i < size; i++){
        q2_vals->set_value(i, xvals_50[i]);
        q2_vals->set_weight(i, wvals_50[i] * exp(xvals_50[i]));
    }

    outer_vals = new dep_vars(size);
    inner_vals = new dep_vars*[size];
    q3_vals = new dummy_vars*[size];
    p4_lows = new int[size];
    p4_highs = new int[size];

    double p1_energy = eps->get_value(p1);
    for(int q2 = 0; q2 < size; q2++){
        double q2_energy = q2_vals->get_value(q2);
        double E2 = sqrt(pow(q2_energy, 2) + pow(me_scaled, 2));
        double E3_min = me_scaled;

        if(p1 < me_scaled / 2 && q2_energy > p1_energy){
            E3_min = p1 + E2 - 0.5 * (2 * p1_energy + E2 - q2_energy + pow(me_scaled, 2) / (2 * p1_energy + E2 - q2_energy));
        }

        double E3_max = 0.5 * (2 * p1_energy + E2 + q2_energy + pow(me_scaled, 2) / (2 * p1_energy + E2 + q2_energy)); //E_lim1
        double p4_min = p1_energy + E2 - E3_max;
        double p4_max = p1_energy + E2 - E3_min;

        int p4_low = 0;
        int p4_high = 0;
        while(eps->get_value(p4_low) <= p4_min){
            p4_low++;
            p4_high++;
            if(p4_low >= eps->get_len()){
                break;
            }
        }
        if(p4_high != eps->get_len()){
            while(eps->get_value(p4_high) < p4_max){
                p4_high++;
                if(p4_high >= eps->get_len()){
                    break;
                }
            }
        }
        p4_high--;
        
        p4_lows[q2] = p4_low;
        p4_highs[q2] = p4_high;

        q3_vals[q2] = new dummy_vars(p4_high - p4_low + 3);
        inner_vals[q2] = new dep_vars(p4_high - p4_low + 3);
        if(E3_min <= me_scaled){
            q3_vals[q2]->set_value(0, 0);
        } else {
            q3_vals[q2]->set_value(0, sqrt(pow(E3_min, 2) - pow(me_scaled, 2)));
        }
        q3_vals[q2]->set_value(p4_high - p4_low + 2, sqrt(pow(E3_max,2) - pow(me_scaled, 2)));

        count = p4_low;
        for(int i = 1; i <= p4_high - p4_low + 1; i++){
            double p4_energy = eps->get_value(p4_low + i - 1);
            q3_vals[q2]->set_value(p4_high - p4_low - i + 2, sqrt(pow(p1_energy + E2 - p4_energy, 2) - pow(me_scaled, 2)));
        }

        q3_vals[q2]->set_trap_weights();
    }

    F_values = new double**[6]; 
    for(int i=0; i<6; i++){
        F_values[i] = new double*[size];
        for(int j = 0; j < size; j++){
            F_values[i][j] = new double[eps->get_len() + 1](); 
        }
    }    
}

double nu_e_collision_R1::get_temp_cm(){
    return temp_cm;
}

dummy_vars** nu_e_collision_R1::get_q3(){
    return q3_vals;
}

double nu_e_collision_R1::M1_R1(double q2, double y, int which_term){
    double p1_energy = eps->get_value(p1);
    double E2 = sqrt(pow(q2, 2) + pow(me_scaled, 2));

    double mult1;
    double mult2;
    if(which_term < 2){
        mult1 = pow(2, 3) * pow(_GF_ * (2 * _Weinberg_ + 1), 2);
        mult2 = 2 * _Weinberg_ / (2 * _Weinberg_ + 1);
    } else {
        mult1 = pow(2, 3) * pow(_GF_ * (2 * _Weinberg_ - 1), 2);
        mult2 = 2 * _Weinberg_ / (2 * _Weinberg_ - 1);
    }
    double mult3 = (1 + mult2) * pow(me_scaled, 2) - pow(E2, 2) - 2 * E2 * p1_energy - pow(p1_energy, 2);
    double mult4 = pow(me_scaled, 2) - pow(E2, 2) - 2 * E2 * p1_energy - pow(p1_energy, 2);
    double mult5 = (1 + 2 * mult2) * pow(me_scaled, 2) - pow(E2, 2) - 2 * E2 * p1_energy - pow(p1_energy, 2);

    double term1 = pow(y, 5) / 5;
    double term2 = (2. / 3) * pow(y, 3) * mult3;
    double term3 = y * mult4 * mult5;

    return mult1 * (term1 + term2 + term3);
}

double nu_e_collision_R1::M2_R1(double q2, double y, int which_term){
    double p1_energy = eps->get_value(p1);
    double E2 = sqrt(pow(q2, 2) + pow(me_scaled, 2));

    double mult1 = pow(2, 5) * pow(_GF_ * _Weinberg_, 2);
    double mult2;
    if(which_term < 2){
        mult2 = -(2 * _Weinberg_ + 1) / 2 * _Weinberg_ ;
    } else {
        mult2 = -(2 * _Weinberg_ - 1) / 2 * _Weinberg_ ;
    }
    double mult3 = (1 + mult2) * pow(me_scaled, 2) - pow(E2, 2) - 2 * E2 * p1_energy - pow(p1_energy, 2);
    double mult4 = pow(me_scaled, 2) - pow(E2, 2) - 2 * E2 * p1_energy - pow(p1_energy, 2);
    double mult5 = (1 + 2 * mult2) * pow(me_scaled, 2) - pow(E2, 2) - 2 * E2 * p1_energy - pow(p1_energy, 2);

    double term1 = pow(y, 5) / 5;
    double term2 = (2. / 3) * pow(y, 3) * mult3;
    double term3 = y * mult4 * mult5;

    return mult1 * (term1 + term2 + term3); 
}

double nu_e_collision_R1::F_comp(freqs_ntT* input, int which_term, int q2, int q3, int check){
    double p1_energy = eps->get_value(p1);
    double q2_energy = q2_vals->get_value(q2);
    double E2 = sqrt(pow(q2_energy, 2) + pow(me_scaled, 2));
    double q3_energy = q3_vals[q2]->get_value(q3);
    double E3 = sqrt(pow(q3_energy, 2) + pow(me_scaled, 2));
    double p4_energy = p1_energy + E2 - E3;

    int num = eps->get_len();
    double f1 = input->get_value(which_term * num + p1);
    double f2 = 1 / (exp(E2 * temp_cm / input->get_temp()) + 1);
    double f3 = 1 / (exp(E3 * temp_cm / input->get_temp()) + 1);
    double f4 = 0;
    if(q3 != 0 && q3 != q3_vals[q2]->get_len() - 1){
        f4 = input->get_value(which_term * num + p4_highs[q2] - q3 + 1);
    } else {
        // interpolation / extrapolation for p4
        if(eps->get_value(num - 1) >= p4_energy){
            int p4 = 0;
            for(int j = 0; j < num; j++){
                if(eps->get_value(j) < p4_energy){
                    p4++;
                }
            }

            int ids[4] = {which_term * num + p4 - 2, which_term * num + p4 - 1, which_term * num + p4, which_term * num + p4 + 1};
            
            if(p4 + 1 >= num){
                ids[3] = which_term * num + p4 - 3;
            }
            if(p4 - 2 < 0){
                ids[0] = which_term * num + p4 + 2;
                if(p4 - 1 < 0){
                    ids[1] = which_term * num + p4 + 3;
                }
            }
            
            
            for(int i = 0; i < 4; i++){
                double multiplier = 1;
                int old_idx = ids[i] % num;
                double old_x_val = eps->get_value(old_idx);
                double old_val = input->get_value(ids[i]);
                for(int j = 0; j < 4; j++){
                    int old_idx2 = ids[j] % num;
                    double mult_x = eps->get_value(old_idx2);
                    if(i != j){
                        multiplier *= (p4_energy - mult_x) / (old_x_val - mult_x);
                    }
                }
                f4 += multiplier * log10(old_val);
            }
            f4 = pow(10, f4); 
        } else {
            double old_eps1 = eps->get_value(num - 2);
            double old_f1 = input->get_value((which_term + 1) * num - 2);
            double old_eps2 = eps->get_value(num - 1);
            double old_f2 = input->get_value((which_term + 1) * num - 1);

            double logy = ((p4_energy - old_eps1) * (log(old_f2) - log(old_f1)) / (old_eps2 - old_eps1)) + log(old_f1);
            f4 = exp(logy);
        }
    }
    return ((check + 1) * f3 * f4 * (1 - f1) * (1 - f2) - check * f1 * f2 * (1 - f3) * (1 - f4));
}

void nu_e_collision_R1::populate_F(freqs_ntT* input, int check){
    int num = eps->get_len();
    int size = q2_vals->get_len();
    for(int i = 0; i < size; i++){
        int num3 = q3_vals[i]->get_len();
        for(int j = 0; j < num + 1; j++){
            for(int k = 0; k < 6; k++){
                if(j < num3){
                    F_values[k][i][j] = F_comp(input, k, i, j, check);
                } else {
                    F_values[k][i][j] = 0;
                }
            }
        }
    }
}

void nu_e_collision_R1::print_F(int i){
    for(int j = 0; j < q3_vals[i]->get_len(); j++){
        cout <<j<<", "<< q3_vals[i]->get_value(j) << ", " << F_values[0][i][j] << endl;
    }
}

double nu_e_collision_R1::interior_integral_R1(int q2, int which_term){
    double p1_energy = eps->get_value(p1);
    double q2_energy = q2_vals->get_value(q2);
    double E2 = sqrt(pow(q2_energy, 2) + pow(me_scaled, 2));
    
    double p_cut1 = sqrt(pow(me_scaled + 2 * pow(p1_energy, 2) / (me_scaled - 2 * p1_energy), 2) - pow(me_scaled, 2));
    double p_cut3 = p1_energy;
    double p_trans2 = sqrt(pow(0.5 * (2 * p1_energy + E2 - q2_energy + pow(me_scaled, 2) / (2 * p1_energy + E2 - q2_energy)), 2) - pow(me_scaled, 2));
    double p_lim1 = sqrt(pow(0.5 * (2 * p1_energy + E2 + q2_energy + pow(me_scaled, 2) / (2 * p1_energy + E2 + q2_energy)), 2) - pow(me_scaled, 2));
    double p_lim2 = p_trans2;

    double M1_2 = M1_R1(q2_energy, p1_energy + q2_energy, which_term) - M1_R1(q2_energy, p1_energy - q2_energy, which_term);
    double M2_2 = M1_R1(q2_energy, p1_energy + q2_energy, which_term) - M1_R1(q2_energy, p1_energy - q2_energy, which_term);

    double q3_energy;
    double E3;
    double yi;
    double yf;

    if(p1_energy < me_scaled / 2){
        if(q2_energy < p_cut3){
            int q3 = 0;
            while(q3_vals[q2]->get_value(q3) < q2_energy){
                q3_energy = q3_vals[q2]->get_value(q3);
                E3 = sqrt(pow(q3_energy, 2) + pow(me_scaled, 2));
                yi = p1_energy + E2 + - E3 - q3_energy;
                yf = p1_energy + E2 - E3 + q3_energy;
                inner_vals[q2]->set_value(q3, (q3_energy / E3) * F_values[which_term][q2][q3] * (M1_R1(q2_energy, yf, which_term) - M1_R1(q2_energy, yi, which_term) + M2_R1(q2_energy, yf, which_term) - M2_R1(q2_energy, yi, which_term)));
                q3++;
            }
            while(q3_vals[q2]->get_value(q3) < p_trans2){
                q3_energy = q3_vals[q2]->get_value(q3);
                E3 = sqrt(pow(q3_energy, 2) + pow(me_scaled, 2));
                inner_vals[q2]->set_value(q3, (q3_energy / E3) * F_values[which_term][q2][q3] * M1_2);
                q3++;
            }
            while(q3 < inner_vals[q2]->length()){
                q3_energy = q3_vals[q2]->get_value(q3);
                E3 = sqrt(pow(q3_energy, 2) + pow(me_scaled, 2));
                yi = E3 + q3_energy - p1_energy - E2;
                yf = p1_energy + q2_energy;
                inner_vals[q2]->set_value(q3, (q3_energy / E3) * F_values[which_term][q2][q3] * (M1_R1(q2_energy, yf, which_term) - M1_R1(q2_energy, yi, which_term) + M2_R1(q2_energy, yf, which_term) - M2_R1(q2_energy, yi, which_term)));
                q3++;
            }
        } else if(q2_energy < p_cut1){
            int q3 = 0;
            while(q3_vals[q2]->get_value(q3) < p_trans2){
                q3_energy = q3_vals[q2]->get_value(q3);
                E3 = sqrt(pow(q3_energy, 2) + pow(me_scaled, 2));
                yi = p1_energy + E2 + - E3 - q3_energy;
                yf = p1_energy + E2 - E3 + q3_energy;
                inner_vals[q2]->set_value(q3, (q3_energy / E3) * F_values[which_term][q2][q3] * (M1_R1(q2_energy, yf, which_term) - M1_R1(q2_energy, yi, which_term) + M2_R1(q2_energy, yf, which_term) - M2_R1(q2_energy, yi, which_term)));
                q3++;
            }
            while(q3_vals[q2]->get_value(q3) < q2_energy){
                q3_energy = q3_vals[q2]->get_value(q3);
                E3 = sqrt(pow(q3_energy, 2) + pow(me_scaled, 2));
                yi = q2_energy - p1_energy;
                yf = p1_energy + E2 - E3 + q3_energy;
                inner_vals[q2]->set_value(q3, (q3_energy / E3) * F_values[which_term][q2][q3] * (M1_R1(q2_energy, yf, which_term) - M1_R1(q2_energy, yi, which_term) + M2_R1(q2_energy, yf, which_term) - M2_R1(q2_energy, yi, which_term)));
                q3++;
            }
            while(q3 < inner_vals[q2]->length()){
                q3_energy = q3_vals[q2]->get_value(q3);
                E3 = sqrt(pow(q3_energy, 2) + pow(me_scaled, 2));
                yi = E3 + q3_energy - p1_energy - E2;
                yf = p1_energy + q2_energy;
                inner_vals[q2]->set_value(q3, (q3_energy / E3) * F_values[which_term][q2][q3] * (M1_R1(q2_energy, yf, which_term) - M1_R1(q2_energy, yi, which_term) + M2_R1(q2_energy, yf, which_term) - M2_R1(q2_energy, yi, which_term)));
                q3++;
            }
        } else {
            int q3 = 0;
            while(q3_vals[q2]->get_value(q3) < q2_energy){
                q3_energy = q3_vals[q2]->get_value(q3);
                E3 = sqrt(pow(q3_energy, 2) + pow(me_scaled, 2));
                yi = q2_energy - p1_energy;
                yf = p1_energy + E2 - E3 + q3_energy;
                inner_vals[q2]->set_value(q3, (q3_energy / E3) * F_values[which_term][q2][q3] * (M1_R1(q2_energy, yf, which_term) - M1_R1(q2_energy, yi, which_term) + M2_R1(q2_energy, yf, which_term) - M2_R1(q2_energy, yi, which_term)));
                q3++;
            }
            while(q3 < inner_vals[q2]->length()){
                q3_energy = q3_vals[q2]->get_value(q3);
                E3 = sqrt(pow(q3_energy, 2) + pow(me_scaled, 2));
                yi = E3 + q3_energy - p1_energy - E2;
                yf = p1_energy + q2_energy;
                inner_vals[q2]->set_value(q3, (q3_energy / E3) * F_values[which_term][q2][q3] * (M1_R1(q2_energy, yf, which_term) - M1_R1(q2_energy, yi, which_term) + M2_R1(q2_energy, yf, which_term) - M2_R1(q2_energy, yi, which_term)));
                q3++;
            }
        }
    } else {
        if(q2_energy < p_cut3){
            int q3 = 0;
            while(q3_vals[q2]->get_value(q3) < q2_energy){
                q3_energy = q3_vals[q2]->get_value(q3);
                E3 = sqrt(pow(q3_energy, 2) + pow(me_scaled, 2));
                yi = p1_energy + E2 + - E3 - q3_energy;
                yf = p1_energy + E2 - E3 + q3_energy;
                inner_vals[q2]->set_value(q3, (q3_energy / E3) * F_values[which_term][q2][q3] * (M1_R1(q2_energy, yf, which_term) - M1_R1(q2_energy, yi, which_term) + M2_R1(q2_energy, yf, which_term) - M2_R1(q2_energy, yi, which_term)));
                q3++;
            }
            while(q3_vals[q2]->get_value(q3) < p_trans2){
                q3_energy = q3_vals[q2]->get_value(q3);
                E3 = sqrt(pow(q3_energy, 2) + pow(me_scaled, 2));
                inner_vals[q2]->set_value(q3, (q3_energy / E3) * F_values[which_term][q2][q3] * M1_2);
                q3++;
            }
            while(q3 < inner_vals[q2]->length()){
                q3_energy = q3_vals[q2]->get_value(q3);
                E3 = sqrt(pow(q3_energy, 2) + pow(me_scaled, 2));
                yi = E3 + q3_energy - p1_energy - E2;
                yf = p1_energy + q2_energy;
                inner_vals[q2]->set_value(q3, (q3_energy / E3) * F_values[which_term][q2][q3] * (M1_R1(q2_energy, yf, which_term) - M1_R1(q2_energy, yi, which_term) + M2_R1(q2_energy, yf, which_term) - M2_R1(q2_energy, yi, which_term)));
                q3++;
            }
        } else {
            int q3 = 0;
            while(q3_vals[q2]->get_value(q3) < p_trans2){
                q3_energy = q3_vals[q2]->get_value(q3);
                E3 = sqrt(pow(q3_energy, 2) + pow(me_scaled, 2));
                yi = p1_energy + E2 + - E3 - q3_energy;
                yf = p1_energy + E2 - E3 + q3_energy;
                inner_vals[q2]->set_value(q3, (q3_energy / E3) * F_values[which_term][q2][q3] * (M1_R1(q2_energy, yf, which_term) - M1_R1(q2_energy, yi, which_term) + M2_R1(q2_energy, yf, which_term) - M2_R1(q2_energy, yi, which_term)));
                q3++;
            }
            while(q3_vals[q2]->get_value(q3) < q2_energy){
                q3_energy = q3_vals[q2]->get_value(q3);
                E3 = sqrt(pow(q3_energy, 2) + pow(me_scaled, 2));
                yi = q2_energy - p1_energy;
                yf = p1_energy + E2 - E3 + q3_energy;
                inner_vals[q2]->set_value(q3, (q3_energy / E3) * F_values[which_term][q2][q3] * (M1_R1(q2_energy, yf, which_term) - M1_R1(q2_energy, yi, which_term) + M2_R1(q2_energy, yf, which_term) - M2_R1(q2_energy, yi, which_term)));
                q3++;
            }
            while(q3 < inner_vals[q2]->length()){
                q3_energy = q3_vals[q2]->get_value(q3);
                E3 = sqrt(pow(q3_energy, 2) + pow(me_scaled, 2));
                yi = E3 + q3_energy - p1_energy - E2;
                yf = p1_energy + q2_energy;
                inner_vals[q2]->set_value(q3, (q3_energy / E3) * F_values[which_term][q2][q3] * (M1_R1(q2_energy, yf, which_term) - M1_R1(q2_energy, yi, which_term) + M2_R1(q2_energy, yf, which_term) - M2_R1(q2_energy, yi, which_term)));
                q3++;
            }
        }
    }
    return q3_vals[q2]->integrate(inner_vals[q2]);
}

void nu_e_collision_R1::whole_integral(freqs_ntT* input, double check, double* results){
    //populates F_values
    this->populate_F(input, check);
    double p_1_energy = eps->get_value(p1);
    for(int i=0; i<6; i++){
        for(int q2=0; q2<q2_vals->get_len(); q2++){
            double q2_energy = q2_vals->get_value(q2);
            double E2 = sqrt(pow(q2_energy, 2) + pow(me_scaled, 2));
            outer_vals->set_value(q2, (q2_energy / E2) * interior_integral_R1(q2, i));
        }
        results[i] = q2_vals->integrate(outer_vals);
        results[i] *= pow(temp_cm, 5) / (pow(2, 4) * pow(2 * _PI_, 3) * pow(p_1_energy, 2));
    }
}

// nu_e_collision_R2 to take care of neutrino-electron R2 collision integrals
nu_e_collision_R2::nu_e_collision_R2(gel_linspace_gl* e, int p1_index, double a){
    scaled_me = _electron_mass_ * a;
    temp_cm = 1./a;
    eps = new gel_linspace_gl(e);
    p1 = p1_index;
    p1_energy = eps->get_value(p1);
    p1_me = p1_energy / scaled_me;
    
    int numgl_points = 50;
    outer_vals = new dep_vars(numgl_points);
    q3_vals = new dummy_vars(numgl_points);
    for(int i=0; i<numgl_points; i++){
        q3_vals->set_value(i, xvals_50[i]);
        q3_vals->set_weight(i, wvals_50[i]*exp(xvals_50[i]));
    }
    
    double E_cut_1 = p1_energy + pow(scaled_me,2)/(4*p1_energy);
    double E_cut_2 = p1_energy + scaled_me*(p1_energy+scaled_me)/(2*p1_energy+scaled_me);
    double E_cut_3 = sqrt(pow(p1_energy,2) + pow(scaled_me,2));
    q_cut_1 = sqrt(pow(E_cut_1,2) - pow(scaled_me,2));
    q_cut_2 = sqrt(pow(E_cut_2,2) - pow(scaled_me,2));
    q_cut_3 = sqrt(pow(E_cut_3,2) - pow(scaled_me,2));
    
    q_trans_2 = new dep_vars(q3_vals->get_len());
    q_lim_1 = new dep_vars(q3_vals->get_len());
    double q3 = 0;
    double E3_energy = 0;
    double E_lim_1 = 0;
    double E_trans_2;
    for(int i=0; i<q3_vals->get_len(); i++){
        q3 = q3_vals->get_value(i);
        E3_energy = sqrt(pow(q3,2) + pow(scaled_me,2));
        E_trans_2 = 0.5 * (E3_energy + q3 - 2*p1_energy + pow(scaled_me,2) / (E3_energy + q3 - 2*p1_energy));
        E_lim_1 = 0.5 * (E3_energy - q3 - 2*p1_energy + pow(scaled_me,2) / (E3_energy - q3 - 2*p1_energy));
        
        q_trans_2->set_value(i, sqrt(pow(E_trans_2,2) - pow(scaled_me,2)));
        q_lim_1->set_value(i, sqrt(pow(E_lim_1,2) - pow(scaled_me,2)));
    }
    
    inner_vals = new dep_vars*[outer_vals->length()];
    q2_vals = new dummy_vars*[q3_vals->get_len()];
    
    double p4_min = 0;
    double p4_max = 0;
    int count_min = 0;
    int count_max = 0;
    double q3_energy = 0;
    double q2_min = 0;
    double q2_max = 0;
    p4_lows = new int[numgl_points];
    p4_highs = new int[numgl_points];
    for(int i=0; i<q3_vals->get_len(); i++){
        q3 = q3_vals->get_value(i);
        E3_energy = sqrt(pow(q3, 2) + pow(scaled_me, 2));
        //idea here is to establish what we want p4 vals to be and then use those vals to reconstruct E3 vals
        
        //first we decide minimum and maximum p4 values. these will be only interpolated p4 values
        //we have to consider cases 
  
        //case 1
        if(p1_me < (sqrt(5)-1)/4){
            //case 1a: m_e < E3 < E_cut_2 => m_e < E2 < E_lim_1
            if(E3_energy < E_cut_2){
                q2_min = 0;
                q2_max = q_lim_1->get_value(i);
            }
            else{
                q2_min = q_trans_2->get_value(i);
                //case 1b: E_cut_2 < E3 < E_cut_1 => E_lim_2 < E2 < E_lim_1                  
                if(E3_energy < E_cut_1){
                    q2_max = q_lim_1->get_value(i);
                }
                //case 1c: E_cut_1 < E3 < inf => E_lim_2 < E2 < inf
                else{
                    q2_max = q3_vals->get_value(numgl_points-1);
                } 
            }
        }
        //case 2
        else if(p1_me < 1/(2*sqrt(2))){
            if(E3_energy < E_cut_2){
                q2_min = 0;
                
                //case 2a: m_e < E3 < E_cut_1 => m_e < E2 < E_lim_1
                if(E3_energy < E_cut_1){
                    q2_max = q_lim_1->get_value(i);
                }
                //case 2b: m_e < E3 < inf => m_e < E2 < inf
                else{
                    q2_max = q3_vals->get_value(numgl_points-1);
                }
                
            }
            //case 2c: E_cut_2 < E3 < inf => E_lim_2 < E2 < inf
            else{
                q2_min = q_trans_2->get_value(i);
                q2_max = q3_vals->get_value(numgl_points-1);
            }
        }
        
        //case 3
        else if(p1_me < 1./2){
            if(E3_energy < E_cut_2){   
                q2_min = 0;
                //case 3a: m_e < E3 < E_cut_1 => m_e < E2 < E_lim_1
                if(E3_energy < E_cut_1){
                    q2_max = q_lim_1->get_value(i);
                }
                //case 3b: E_cut_1 < E3 < E_cut_2 => m_e < E2 < inf
                else{
                    q2_max = q3_vals->get_value(numgl_points-1);
                }
            }
            //case 3c: E_cut_2 < E3 < inf => E_lim_2 < E2 < inf
            else{
                q2_min = q_trans_2->get_value(i);
                q2_max = q3_vals->get_value(numgl_points-1);
            }
        }
        //case 4
        else{
            q2_max = q3_vals->get_value(numgl_points-1);
            //case 4a: m_e < E3 < E_cut_2 => m_e < E2 < inf
            if(E3_energy < E_cut_2){
                q2_min = 0;
            }
            //case 4b: E_cut_2 < E3 < inf => E_lim_2 < E2 < inf
            else{
                q2_min = q_trans_2->get_value(i);
            }
        }
        p4_min = p1_energy + sqrt(pow(q2_min, 2) + pow(scaled_me, 2)) - E3_energy;
        p4_max = p1_energy + sqrt(pow(q2_max, 2) + pow(scaled_me, 2)) - E3_energy;
        
        double temp_energy = eps->get_value(0);
        count_min = 0;
        count_max = 0;
        //count_min gives the number of items in epsilon that have energy less than the minimum p4 val; therefore first p4 val of interest is epsilon[count_min]
        while(temp_energy < p4_min){
            count_min++;
            if(count_min >= eps->get_len()){
                break;
            }
            temp_energy = eps->get_value(count_min);
        }
        
        count_max = count_min;
        
        if(count_min != eps->get_len()){
            while(temp_energy < p4_max){
                count_max++;
                if(count_max >= eps->get_len()){
                    break;
                }
                temp_energy = eps->get_value(count_max);
            }
        }
        count_max--;
        
        p4_lows[i] = count_min;
        p4_highs[i] = count_max;
        //count_max gives the index of the greatest element of epsilon that has energy less than the maximum p4 val; therefore last p4 val of interest is epsilon[count_max]
        
        //p4 vals will contain p4_min, epsilon values from indices count_min to count_max, inclusive, and p4_max
        //therefore E2_vals needs to have 3+count_max-count_min things in it
        //note that if count_min and count_max beyond the end of the array, count_max=count_min-1 so q2_vals will have length 2
        q2_vals[i] = new dummy_vars(count_max-count_min+3);
        q2_vals[i]->set_value(0, q2_min);
        q2_vals[i]->set_value(count_max-count_min+2, q2_max);
        
        for(int j=count_min; j<=count_max; j++){
            //q2 = p4 - p1 + q3
            q2_vals[i]->set_value(j-count_min+1, sqrt(pow(eps->get_value(j) - p1_energy + E3_energy, 2) - pow(scaled_me, 2)));
        }
        
        q2_vals[i]->set_trap_weights();
        
        inner_vals[i] = new dep_vars(count_max-count_min+3);
    }
    
    F_values = new double**[6]();
    for(int i=0; i<6; i++){
        F_values[i] = new double*[eps->get_len()+1]();
        for(int j=0; j<eps->get_len() + 2; j++){
            F_values[i][j] = new double[q3_vals->get_len()];
        }
    } 
    
}

nu_e_collision_R2::~nu_e_collision_R2(){
    for(int i=0; i<6; i++){
        for(int j=0; j<eps->get_len()+1; j++){
            delete[] F_values[i][j];
        }
        delete[] F_values[i];
    }
    delete[] F_values;
    
    for(int i=0; i<q3_vals->get_len(); i++){
        delete q2_vals[i];
        delete inner_vals[i];
    }
    delete q2_vals;
    delete inner_vals;
    
    delete outer_vals;
    delete q3_vals;
    
    
    delete eps;
    delete q_trans_2;
    delete q_lim_1;   
}

double nu_e_collision_R2::F_comp(freqs_ntT* input, int which_term, int q2, int q3, int check){
    double p1_energy = eps->get_value(p1);
    double q3_energy = q3_vals->get_value(q3);
    double E3 = sqrt(pow(q3_energy, 2) + pow(scaled_me, 2));
    double q2_energy = q2_vals[q3]->get_value(q2);
    double E2 = sqrt(pow(q2_energy, 2) + pow(scaled_me, 2));
    double p4_energy = p1_energy + E2 - E3;

    int num = eps->get_len();
    double f1 = input->get_value(which_term * num + p1);
    double f2 = 1 / (exp(E2 * temp_cm / input->get_temp()) + 1);
    double f3 = 1 / (exp(E3 * temp_cm / input->get_temp()) + 1);
    double f4 = 0;
    if(q2 != 0 && q2 != q2_vals[q3]->get_len() - 1){
        f4 = input->get_value(which_term * num + p4_lows[q3] + q2 - 1);
    } else {
        // interpolation / extrapolation for p4
        if(eps->get_value(num - 1) >= p4_energy){
            int p4 = 0;
            for(int j = 0; j < num; j++){
                if(eps->get_value(j) < p4_energy){
                    p4++;
                }
            }

            int ids[4] = {which_term * num + p4 - 2, which_term * num + p4 - 1, which_term * num + p4, which_term * num + p4 + 1};
            
            if(p4 + 1 >= num){
                ids[3] = which_term * num + p4 - 3;
            }
            if(p4 - 2 < 0){
                ids[0] = which_term * num + p4 + 2;
                if(p4 - 1 < 0){
                    ids[1] = which_term * num + p4 + 3;
                }
            }
            
            
            for(int i = 0; i < 4; i++){
                double multiplier = 1;
                int old_idx = ids[i] % num;
                double old_x_val = eps->get_value(old_idx);
                double old_val = input->get_value(ids[i]);
                for(int j = 0; j < 4; j++){
                    int old_idx2 = ids[j] % num;
                    double mult_x = eps->get_value(old_idx2);
                    if(i != j){
                        multiplier *= (p4_energy - mult_x) / (old_x_val - mult_x);
                    }
                }
                f4 += multiplier * log10(old_val);
            }
            f4 = pow(10, f4); 
        } else {
            double old_eps1 = eps->get_value(num - 2);
            double old_f1 = input->get_value((which_term + 1) * num - 2);
            double old_eps2 = eps->get_value(num - 1);
            double old_f2 = input->get_value((which_term + 1) * num - 1);

            double logy = ((p4_energy - old_eps1) * (log(old_f2) - log(old_f1)) / (old_eps2 - old_eps1)) + log(old_f1);
            f4 = exp(logy);
        }
    }
    return ((check + 1) * f3 * f4 * (1 - f1) * (1 - f2) - check * f1 * f2 * (1 - f3) * (1 - f4));
}

void nu_e_collision_R2::populate_F(freqs_ntT* input, int check){
    int num = eps->get_len();
    int size = q3_vals->get_len();
    for(int i = 0; i < num + 2; i++){
        for(int j = 0; j < size; j++){
            int num3 = q2_vals[j]->get_len();
            for(int k = 0; k < 6; k++){
                if(i < num3){
                    F_values[k][i][j] = F_comp(input, k, i, j, check);
                } else {
                    F_values[k][i][j] = 0;
                }
            }
        }
    }
}

double nu_e_collision_R2::integrated_M_1_prime(double y, double E3, int which_term){
    double me_square = pow(scaled_me, 2);
    double mult1;
    double mult2;
    if(which_term < 2){
        mult1 = pow(2, 3) * pow(_GF_ * (2 * _Weinberg_ + 1), 2);
        mult2 = 2 * _Weinberg_ / (2 * _Weinberg_ + 1);
    } else {
        mult1 = pow(2, 3) * pow(_GF_ * (2 * _Weinberg_ - 1), 2);
        mult2 = 2 * _Weinberg_ / (2 * _Weinberg_ - 1);
    }
    double result = (pow(p1_energy-E3,2) - pow(scaled_me,2)) * (pow(p1_energy-E3,2) + (2 * mult2 - 1)*pow(scaled_me,2)) * y;
    result -= (2./3) *  (pow(p1_energy-E3,2) + (mult2 - 1) * pow(scaled_me,2)) * pow(y,3);
    result += pow(y,5)/5;
    result *= mult1;

    return result;
}

double nu_e_collision_R2::M_1_1(double q2, double E2, double E3, int which_term){
    double lower_bound = p1_energy - E3 + E2 - q2;
    double upper_bound = p1_energy - E3 + E2 + q2;
    
    return integrated_M_1_prime(upper_bound, E3, which_term) - integrated_M_1_prime(lower_bound, E3, which_term);
}

double nu_e_collision_R2::M_1_2(double q3, double E3, int which_term){
    double lower_bound = p1_energy - q3;
    double upper_bound = p1_energy + q3;
    
    return integrated_M_1_prime(upper_bound, E3, which_term) - integrated_M_1_prime(lower_bound, E3, which_term);
}

double nu_e_collision_R2::M_1_3(double q2, double E2, double q3, double E3, int which_term){
    double lower_bound = E3 - p1_energy - E2 + q2;
    double upper_bound = p1_energy + q3;
    
    return integrated_M_1_prime(upper_bound, E3, which_term) - integrated_M_1_prime(lower_bound, E3, which_term);
}

double nu_e_collision_R2::M_1_4(double q2, double E2, double q3, double E3, int which_term){
    double lower_bound = q3 - p1_energy;
    double upper_bound = p1_energy - E3 + E2 + q2;
    
    return integrated_M_1_prime(upper_bound, E3, which_term) - integrated_M_1_prime(lower_bound, E3, which_term);
}

double nu_e_collision_R2::integrated_M_2_prime(double y, double E3, int which_term){
    double me_square = pow(scaled_me, 2);
    double mult1 = pow(2,5) * pow(_GF_,2) * pow(_Weinberg_,2);
    double mult2;
    if(which_term < 2){
        mult2 = - (2 * _Weinberg_ + 1) / (2 * _Weinberg_);
    } else {
        mult2 = - (2 * _Weinberg_ - 1) / (2 * _Weinberg_);
    }
    double result = (pow(p1_energy-E3,2) - pow(scaled_me,2)) * (pow(p1_energy-E3,2) + (2 * mult2 - 1)*pow(scaled_me,2)) * y;
    result -= (2./3) *  (pow(p1_energy-E3,2) + (mult2 - 1) * pow(scaled_me,2)) * pow(y,3);
    result += pow(y,5)/5;
    result *= mult1;
    return result;
}

double nu_e_collision_R2::M_2_1(double q2, double E2, double E3, int which_term){
    double lower_bound = p1_energy - E3 + E2 - q2;
    double upper_bound = p1_energy - E3 + E2 + q2;
    
    return integrated_M_2_prime(upper_bound, E3, which_term) - integrated_M_2_prime(lower_bound, E3, which_term);
}

double nu_e_collision_R2::M_2_2(double q3, double E3, int which_term){
    double lower_bound = p1_energy - q3;
    double upper_bound = p1_energy + q3;
    
    return integrated_M_2_prime(upper_bound, E3, which_term) - integrated_M_2_prime(lower_bound, E3, which_term);
}

double nu_e_collision_R2::M_2_3(double q2, double E2, double q3, double E3, int which_term){
    double lower_bound = E3 - p1_energy - E2 + q2;
    double upper_bound = p1_energy + q3;
    
    return integrated_M_2_prime(upper_bound, E3, which_term) - integrated_M_2_prime(lower_bound, E3, which_term);
}

double nu_e_collision_R2::M_2_4(double q2, double E2, double q3, double E3, int which_term){
    double lower_bound = q3 - p1_energy;
    double upper_bound = p1_energy - E3 + E2 + q2;
    
    return integrated_M_2_prime(upper_bound, E3, which_term) - integrated_M_2_prime(lower_bound, E3, which_term);
}

double nu_e_collision_R2::inner_integral(int which_term, int q3){
    double q3_momentum = q3_vals->get_value(q3);
    double E3 = sqrt(pow(q3_momentum,2) + pow(scaled_me,2));
    //case 1
    if(p1_me < (sqrt(5)-1)/4.){
        //case 1a
        if(q3_momentum < q_cut_3){
            for(int q2=0; q2<q2_vals[q3]->get_len(); q2++){
                double q2_momentum = q2_vals[q3]->get_value(q2);
                double E2 = sqrt(pow(q2_momentum,2) + pow(scaled_me,2));
                
                //case 1ai
                if(q2_momentum < q3_momentum){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_1(q2_momentum, E2, E3, which_term) + M_1_1(q2_momentum, E2, E3, which_term)));
                }
                //case 1aii
                else if(q2_momentum < q_trans_2->get_value(q3)){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_2(q3_momentum, E3, which_term) + M_1_2(q3_momentum, E3, which_term)));
                }
                //case 1aiii
                else{
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_3(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_3(q2_momentum, E2, q3_momentum, E3, which_term)));
                }    
            }
        }
        //case 1b
        else if(q3_momentum < q_cut_2){
            for(int q2=0; q2<q2_vals[q3]->get_len(); q2++){
                double q2_momentum = q2_vals[q3]->get_value(q2);
                double E2 = sqrt(pow(q2_momentum,2) + pow(scaled_me,2));
                
                //case 1bi
                if(q2_momentum < q_trans_2->get_value(q3)){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_1(q2_momentum, E2, E3, which_term) + M_1_1(q2_momentum, E2, E3, which_term)));
                }
                //case 1bii
                else if(q2_momentum < q3_momentum){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_4(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_4(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                //case 1biii
                else{
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_3(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_3(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                
            }
        }
        //case 1c
        else if(q3_momentum < q_cut_1){
            for(int q2=0; q2<q2_vals[q3]->get_len(); q2++){
                double q2_momentum = q2_vals[q3]->get_value(q2);
                double E2 = sqrt(pow(q2_momentum,2) + pow(scaled_me,2));
                
                //case 1ci
                if(q2_momentum < q3_momentum){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_4(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_4(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                //case 1cii
                else{
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_3(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_3(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                
            }
        }
        //case 1d
        else{
            for(int q2=0; q2<q2_vals[q3]->get_len(); q2++){
                double q2_momentum = q2_vals[q3]->get_value(q2);
                double E2 = sqrt(pow(q2_momentum,2) + pow(scaled_me,2));
                
                //case 1ci
                if(q2_momentum < q3_momentum){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_4(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_4(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                //case 1cii
                else{
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_3(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_3(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                
            }
        }
    }
    
    //case 2
    else if(p1_me < 1./(2 * sqrt(2))){
        //case 2a
        if(q3_momentum<q_cut_3){
            for(int q2=0; q2<q2_vals[q3]->get_len(); q2++){
                double q2_momentum = q2_vals[q3]->get_value(q2);
                double E2 = sqrt(pow(q2_momentum,2) + pow(scaled_me,2));
                
                //case 2ai
                if(q2_momentum < q3_momentum){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_1(q2_momentum, E2, E3, which_term) + M_1_1(q2_momentum, E2, E3, which_term)));
                }
                //case 2aii
                else if(q2_momentum < q_trans_2->get_value(q3)){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_2(q3_momentum, E3, which_term) + M_1_2(q3_momentum, E3, which_term)));
                }
                //case 2aiii
                else{
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_3(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_3(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                
            }
        }
        //case 2b
        else if(q3_momentum<q_cut_1){
            for(int q2=0; q2<q2_vals[q3]->get_len(); q2++){
                double q2_momentum = q2_vals[q3]->get_value(q2);
                double E2 = sqrt(pow(q2_momentum,2) + pow(scaled_me,2));
                
                //case 2bi
                if(q2_momentum < q_trans_2->get_value(q3)){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_1(q2_momentum, E2, E3, which_term) + M_1_1(q2_momentum, E2, E3, which_term)));
                }
                //case 2bii
                else if(q2_momentum < q3_momentum){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_4(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_4(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                //case 2biii
                else{
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_3(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_3(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                
            }
        }
        //case 2c
        else if(q3_momentum<q_cut_2){
            for(int q2=0; q2<q2_vals[q3]->get_len(); q2++){
                double q2_momentum = q2_vals[q3]->get_value(q2);
                double E2 = sqrt(pow(q2_momentum,2) + pow(scaled_me,2));
                
                //case 2ci
                if(q2_momentum < q_trans_2->get_value(q3)){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_1(q2_momentum, E2, E3, which_term) + M_1_1(q2_momentum, E2, E3, which_term)));
                }
                //case 2cii
                else if(q2_momentum < q3_momentum){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_4(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_4(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                //case 2ciii
                else{
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_3(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_3(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                
            }
        }
        //case 2d
        else{
            for(int q2=0; q2<q2_vals[q3]->get_len(); q2++){
                double q2_momentum = q2_vals[q3]->get_value(q2);
                double E2 = sqrt(pow(q2_momentum,2) + pow(scaled_me,2));
                
                //case 2di
                if(q2_momentum < q3_momentum){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_4(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_4(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                //case 2dii
                else{
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_3(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_3(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                
            }
        }
        
    }
    
    //case 3
    else if(p1_me < 0.5){
        //case 3a
        if(q3_momentum<q_cut_1){
            for(int q2=0; q2<q2_vals[q3]->get_len(); q2++){
                double q2_momentum = q2_vals[q3]->get_value(q2);
                double E2 = sqrt(pow(q2_momentum,2) + pow(scaled_me,2));
                
                //case 3ai
                if(q2_momentum < q3_momentum){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_1(q2_momentum, E2, E3, which_term) + M_1_1(q2_momentum, E2, E3, which_term)));
                }
                //case 3aii
                else if(q2_momentum < q_trans_2->get_value(q3)){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_2(q3_momentum, E3, which_term) + M_1_2(q3_momentum, E3, which_term)));
                }
                //case 3aiii
                else{
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_3(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_3(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                
            }
        }
        //case 3b
        else if(q3_momentum<q_cut_3){
            for(int q2=0; q2<q2_vals[q3]->get_len(); q2++){
                double q2_momentum = q2_vals[q3]->get_value(q2);
                double E2 = sqrt(pow(q2_momentum,2) + pow(scaled_me,2));
                
                //case 3bi
                if(q2_momentum < q3_momentum){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_1(q2_momentum, E2, E3, which_term) + M_1_1(q2_momentum, E2, E3, which_term)));
                }
                //case 3bii
                else if(q2_momentum < q_trans_2->get_value(q3)){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_2(q3_momentum, E3, which_term) + M_1_2(q3_momentum, E3, which_term)));
                }
                //case 3biii
                else{
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_3(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_3(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                
            }
        }
        //case 3c
        else if(q3_momentum<q_cut_2){
            for(int q2=0; q2<q2_vals[q3]->get_len(); q2++){
                double q2_momentum = q2_vals[q3]->get_value(q2);
                double E2 = sqrt(pow(q2_momentum,2) + pow(scaled_me,2));
                
                //case 3ci
                if(q2_momentum < q_trans_2->get_value(q3)){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_1(q2_momentum, E2, E3, which_term) + M_1_1(q2_momentum, E2, E3, which_term)));
                }
                //case 3cii
                else if(q2_momentum < q3_momentum){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_4(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_4(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                //case 3ciii
                else{
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_3(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_3(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                
            }
        }
        //case 3d
        else{
            for(int q2=0; q2<q2_vals[q3]->get_len(); q2++){
                double q2_momentum = q2_vals[q3]->get_value(q2);
                double E2 = sqrt(pow(q2_momentum,2) + pow(scaled_me,2));
                
                //case 3di
                if(q2_momentum < q3_momentum){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_4(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_4(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                //case 3dii
                else{
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_3(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_3(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                
            }
        }        
    }
    
    //case 4
    else{
         //case 4a
        if(q3_momentum<q_cut_1){
            for(int q2=0; q2<q2_vals[q3]->get_len(); q2++){
                double q2_momentum = q2_vals[q3]->get_value(q2);
                double E2 = sqrt(pow(q2_momentum,2) + pow(scaled_me,2));
                
                //case 4ai
                if(q2_momentum < q3_momentum){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_1(q2_momentum, E2, E3, which_term) + M_1_1(q2_momentum, E2, E3, which_term)));
                }
                //case 4aii
                else{
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_2(q3_momentum, E3, which_term) + M_1_2(q3_momentum, E3, which_term)));
                }
                
            } 
        }
        //case 4b
        else if(q3_momentum<q_cut_3){
            for(int q2=0; q2<q2_vals[q3]->get_len(); q2++){
                double q2_momentum = q2_vals[q3]->get_value(q2);
                double E2 = sqrt(pow(q2_momentum,2) + pow(scaled_me,2));
                
                //case 4bi
                if(q2_momentum < q3_momentum){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_1(q2_momentum, E2, E3, which_term) + M_1_1(q2_momentum, E2, E3, which_term)));
                }
                //case 4bii
                else if(q2_momentum < q_trans_2->get_value(q3)){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_2(q3_momentum, E3, which_term) + M_1_2(q3_momentum, E3, which_term)));
                }
                //case 4biii
                else{
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_3(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_3(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                
            }
        }
        //case 4c
        else if(q3_momentum<q_cut_2){
            for(int q2=0; q2<q2_vals[q3]->get_len(); q2++){
                double q2_momentum = q2_vals[q3]->get_value(q2);
                double E2 = sqrt(pow(q2_momentum,2) + pow(scaled_me,2));
                
                //case 4ci
                if(q2_momentum < q_trans_2->get_value(q3)){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_1(q2_momentum, E2, E3, which_term) + M_1_1(q2_momentum, E2, E3, which_term)));
                }
                //case 4cii
                else if(q2_momentum < q3_momentum){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_4(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_4(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                //case 4ciii
                else{
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_3(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_3(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                
            }
        }
        //case 4d
        else{
            for(int q2=0; q2<q2_vals[q3]->get_len(); q2++){
                double q2_momentum = q2_vals[q3]->get_value(q2);
                double E2 = sqrt(pow(q2_momentum,2) + pow(scaled_me,2));
                
                //case 4di
                if(q2_momentum < q3_momentum){
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_4(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_4(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
                //case 4dii
                else{
                    inner_vals[q3]->set_value(q2, (q2_momentum / E2) * F_values[which_term][q2][q3] * (M_2_3(q2_momentum, E2, q3_momentum, E3, which_term) + M_1_3(q2_momentum, E2, q3_momentum, E3, which_term)));
                }
            }
            
        }  
        
    }
    double result = q2_vals[q3]->integrate(inner_vals[q3]);
    return result;
}

double nu_e_collision_R2::whole_integral(freqs_ntT* input, double a, double check, double* results){
    if(p1_energy == 0){
        for(int i=0; i<6; i++){
            results[i] = 0;
        }
    } else {
        //populate F values
        this->populate_F(input, check);
        double q3_momentum = 0;
        double E3 = 0;
        for(int i = 0; i < 6; i++){
            for(int q3=0; q3<q3_vals->get_len(); q3++){
                q3_momentum = q3_vals->get_value(q3);
                E3 = sqrt(pow(q3_momentum,2) + pow(scaled_me,2));
                outer_vals->set_value(q3, (q3_momentum / E3) * inner_integral(i, q3));
            }
            results[i] = q3_vals->integrate(outer_vals);
            results[i] *= pow(temp_cm, 5) / (pow(2,4) * pow(2*_PI_,3) * pow(p1_energy,2));
        }
    }
}
