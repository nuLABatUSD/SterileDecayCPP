#include <cmath>
#include "constants.hh"
#include "arrays.hh"
#include "decays.h"
#include <fstream>
using std::cout;
using std::endl;
using std::ofstream;


double get_rate(double ms, double theta, int identifier){
    switch(identifier){
        case 1:
            return 3 * _fine_structure_ * pow(_GF_,2) * pow(ms,5) * pow(sin(theta),2) / (512 * pow(_PI_,4));
        case 2:
            if (ms >= _neutral_pion_mass_){
                return pow(_GF_,2) * pow(_pion_decay_,2) * ms * (pow(ms,2) - pow(_neutral_pion_mass_,2)) * pow(sin(theta),2) / (48 * _PI_);
            } else {
                return 0;
            }
        case 3:
            if (ms >= _charged_pion_mass_ + _electron_mass_){
                return pow(_GF_,2) * pow(_pion_decay_,2) * ms * sqrt((pow(ms,2) - pow(_charged_pion_mass_ + _electron_mass_,2)) * (pow(ms,2) - pow(_charged_pion_mass_ - _electron_mass_,2))) * pow(sin(theta),2) / (16 * _PI_);
            } else { 
                return 0;
            }
        case 4:
            if (ms >= _charged_pion_mass_ + _muon_mass_){
                return pow(_GF_,2) * pow(_pion_decay_,2) * ms * sqrt((pow(ms,2) - pow(_charged_pion_mass_ + _muon_mass_,2)) * (pow(ms,2) - pow(_charged_pion_mass_ - _muon_mass_,2))) * pow(sin(theta),2) / (16 * _PI_);
            } else {
                return 0;
            }
        default:
            cout << "Error: this identifier number is not supported" << endl;
    }
}

double get_lifetime(double ms, double theta){
    double rate_1 = get_rate(ms, theta, 1);
    double rate_2 = get_rate(ms, theta, 2);
    double rate_3 = get_rate(ms, theta, 3);
    double rate_4 = get_rate(ms, theta, 4);
    return 1 / (3 * rate_1 + 3 * rate_2 + 2 * rate_3 + 2 * rate_4);
}

double get_monoenergy(double m0, double m1, double m2){
    return (pow(m0,2) + pow(m1,2) - pow(m2,2)) / (2 * m0);
}

void compute_kinetics(double m0, double m1, double m2, double* gamma, double* v, double* p){
    double particle_energy = get_monoenergy(m0,m1,m2);

    *gamma = particle_energy / m1;
    *p = sqrt(pow(particle_energy,2) - pow(m1,2));
    *v = *(p) / particle_energy;
}

double get_gamma_a(double evaluate_at){
    double term_1 = 3 * pow(_electron_mass_,4);
    double term_2 = 6 * pow(_electron_mass_,2) * _muon_mass_ * evaluate_at;
    double term_3 = pow(_muon_mass_,2) * evaluate_at * (4 * evaluate_at - 3 * _muon_mass_);
    double term_4 = pow(_electron_mass_,4) * _muon_mass_ * log(_muon_mass_ - 2 * evaluate_at);
    return -((evaluate_at / 6) * (term_1 + term_2 + term_3) + term_4 / 4) / pow(_muon_mass_,3);
}

double get_gamma_b(double evaluate_at){
    double term_1 = -16 * pow(_electron_mass_,2) * _muon_mass_ * pow(evaluate_at,3);
    double term_2 = -6 * pow(_electron_mass_,4) * evaluate_at * (_muon_mass_ + evaluate_at);
    double term_3 = -4 * pow(_muon_mass_,2) * pow(evaluate_at,3) * (3 * evaluate_at - 2 * _muon_mass_);
    double term_4 = -3 * pow(_electron_mass_,4) * pow(_muon_mass_,2) * log(_muon_mass_ - 2 * evaluate_at);
    return (term_1 + term_2 + term_3 + term_4) / (24 *pow(_muon_mass_,3));
}

double get_decay_type_one(double decay_rate, double energy, double width, double ms, double product_mass){
    double neutrino_energy = get_monoenergy(ms, 0, product_mass);
    double factor = 0;
    double sigma = 0.05 * width;
    double height = 1 / (width - 2 * sigma);
    double sig2 = 1 / (2 * pow(sigma, 2));
    if(energy <= neutrino_energy){
        factor = 0;
    } else if(energy <= neutrino_energy + sigma){
        factor = pow(energy - neutrino_energy, 2) * height * sig2;
    } else if(energy <= neutrino_energy + 2 * sigma){
        factor = height * (1 - pow(energy - neutrino_energy - 2 * sigma, 2) * sig2);
    } else if(energy <= neutrino_energy + width - 2 * sigma){
        factor = height;
    } else if(energy <= neutrino_energy + width - sigma){
        factor = height * (1 - pow(energy - width + 2 * sigma - neutrino_energy, 2) * sig2);
    } else if(energy <= neutrino_energy + width){
        factor = pow(energy - width - neutrino_energy, 2) * height * sig2;
    }

    return decay_rate * factor; 
    
    /*
    double neutrino_energy = get_monoenergy(ms, 0, product_mass);
    if (energy < neutrino_energy && neutrino_energy <= energy + width){
        return decay_rate / width;
    } else {
        return 0;
    }*/
}

double get_decay_type_two(double decay_rate, double energy, double ms, double spectator_mass){
    double* gamma_pion = new double;
    double* v_pion = new double;
    double* p_pion = new double;

    double neutrino_energy_pion = get_monoenergy(_charged_pion_mass_, 0, _muon_mass_);
    double factor = 0;

    compute_kinetics(ms, _charged_pion_mass_, spectator_mass, gamma_pion, v_pion, p_pion);
    double width = 2 * *gamma_pion * *v_pion * neutrino_energy_pion;
    double min = *gamma_pion * neutrino_energy_pion * (1 - *v_pion);
    double sigma = 0.05 * width;
    double height = 1 / (width - 2 * sigma);
    double sig2 = 1 / (2 * pow(sigma, 2));
    if(energy <= min){
        factor = 0;
    } else if(energy <= min + sigma){
        factor = pow(energy - min, 2) * height * sig2;
    } else if(energy <= min + 2 * sigma){
        factor = height * (1 - pow(energy - min - 2 * sigma, 2) * sig2);
    } else if(energy <= min + width - 2 * sigma){
        factor = height;
    } else if(energy <= min + width - sigma){
        factor = height * (1 - pow(energy - width + 2 * sigma - min, 2) * sig2);
    } else if(energy <= min + width){
        factor = pow(energy - width - min, 2) * height * sig2;
    }

    delete gamma_pion;
    delete v_pion;
    delete p_pion;

    return decay_rate * factor;
    /*
    double* gamma_pion = new double;
    double* v_pion = new double;
    double* p_pion = new double;

    double neutrino_energy_pion = get_monoenergy(_charged_pion_mass_, 0, _muon_mass_);
    double ddecay_rate = 0;

    compute_kinetics(ms, _charged_pion_mass_, spectator_mass, gamma_pion, v_pion, p_pion);

    if (*(gamma_pion) * neutrino_energy_pion * (1 - *(v_pion)) <= energy && energy <= *(gamma_pion) * neutrino_energy_pion * (1 + *(v_pion))){
        ddecay_rate = decay_rate / (2 * *(gamma_pion) * *(v_pion) * neutrino_energy_pion);
    }
    
    delete gamma_pion;
    delete v_pion;
    delete p_pion;

    return ddecay_rate;
    */
}

double get_decay_type_three(double decay_rate, double energy, double ms){
    double* gamma_muon = new double;
    double* v_muon = new double;
    double* p_muon = new double;
    double ddecay_rate = 0;

    double neutrino_max_energy_muon = get_monoenergy(_muon_mass_, 0, _electron_mass_);
    compute_kinetics(ms, _muon_mass_, _charged_pion_mass_, gamma_muon, v_muon, p_muon);

    double a = energy / (*gamma_muon * (1 + *v_muon));
    double b = neutrino_max_energy_muon;

    if (energy / (*gamma_muon * (1 - *v_muon)) < b){
        b = energy / (*gamma_muon * (1 - *v_muon));
    }
    if (a < b){
        ddecay_rate = decay_rate * (get_gamma_a(b) - get_gamma_a(a)) / (2 * *gamma_muon * *v_muon * (get_gamma_b(neutrino_max_energy_muon) - get_gamma_b(0)));
    }
    
    delete gamma_muon;
    delete v_muon;
    delete p_muon;
    
    return ddecay_rate;
}

double type_four_integrand(double energy, double muon_energy){
    double gamma_muon = muon_energy / _muon_mass_;
    double v_muon = sqrt(pow(muon_energy,2) - pow(_muon_mass_,2)) / muon_energy;
    double neutrino_max_energy_muon = get_monoenergy(_muon_mass_, 0, _electron_mass_);
    double a = energy / (gamma_muon * (1 + v_muon));
    double b = neutrino_max_energy_muon;

    if (energy / (gamma_muon * (1 - v_muon)) < b){
        b = energy / (gamma_muon * (1 - v_muon));
    }
    if (b - a > 1e-4){
        return (get_gamma_a(b) - get_gamma_a(a)) / (2 * gamma_muon * v_muon * (get_gamma_b(neutrino_max_energy_muon) - get_gamma_b(0)));
    } else {
        return 0;
    }
}

double get_decay_type_four(double decay_rate, double energy, double ms, double pion_spectator_mass){
    double* gamma_pion = new double;
    double* v_pion = new double;
    double* p_pion = new double;
    double* gamma_muon = new double;
    double* v_muon = new double;
    double* p_muon = new double;

    compute_kinetics(ms, _charged_pion_mass_, pion_spectator_mass, gamma_pion, v_pion, p_pion);
    compute_kinetics(_charged_pion_mass_, _muon_mass_, 0, gamma_muon, v_muon, p_muon);
    
    double start = *gamma_pion * (*gamma_muon * _muon_mass_ - *v_pion * *p_muon);
    double end =  *gamma_pion * (*gamma_muon * _muon_mass_ + *v_pion * *p_muon);
    dep_vars* four_vals = new dep_vars(100);
    gel_dummy_vars* x = new gel_dummy_vars(start, end);

    for (int i = 0; i < 100; i++){
        four_vals->set_value(i,type_four_integrand(energy, ((end - start) * x->get_value(i) + (end + start)) / 2));
    }

    double integral = x->integrate(four_vals);
    double four = decay_rate * integral / (2 * *gamma_pion * *v_pion * *p_muon);

    delete gamma_pion;
    delete v_pion;
    delete p_pion;
    delete gamma_muon;
    delete v_muon;
    delete p_muon;
    delete four_vals;
    delete x;

    return four; 
}

void compute_dPdtdE(linspace_and_gl* energies_cm, double ms, double theta, double temp_cm, dummy_vars* pe, dummy_vars* pae, dummy_vars* pm, dummy_vars* pam, dummy_vars* pt, dummy_vars* pat){
    int num_bins = energies_cm->get_len();

    dummy_vars* energy_bins = new dummy_vars(num_bins);

    for(int i = 0; i < num_bins; i++){
        energy_bins->set_value(i,energies_cm->get_value(i) * temp_cm);
    }

    double bin_width = energy_bins->get_value(1) - energy_bins->get_value(0);

    for(int i = 0; i < num_bins; i++){
        // define decay rates
        double rate_1 = get_rate(ms, theta, 1);
        double rate_2 = get_rate(ms, theta, 2);
        double rate_3 = get_rate(ms, theta, 3);
        double rate_4 = get_rate(ms, theta, 4);


        // defining relevant variables
        double energy = energy_bins->get_value(i);
        double d2 = 0;
        double d3_2 = 0;
        double d3_4 = 0;
        double d4_2 = 0;
        double d4_3 = 0;
        double d4_4 = 0;

        // decay 1 contributions (all identical)
        double d1 = get_decay_type_one(rate_1, energy, bin_width, ms, 0);

        // decay 2 contributions (all identical)
        if(ms >= _neutral_pion_mass_){
            d2 = get_decay_type_one(rate_2, energy, bin_width, ms, _neutral_pion_mass_);
        }

        // decay 3 contributions
        if(ms >= _electron_mass_ + _charged_pion_mass_){
            d3_2 = get_decay_type_two(rate_3, energy, ms, _electron_mass_);
            d3_4 = get_decay_type_four(rate_3, energy, ms, _electron_mass_);
        }

        // decay 4 contributions
        if(ms >= _muon_mass_ + _charged_pion_mass_){
            d4_2 = get_decay_type_two(rate_4, energy, ms, _muon_mass_);
            d4_3 = get_decay_type_three(rate_4, energy, ms);
            d4_4 = get_decay_type_four(rate_4, energy, ms, _muon_mass_);
        }
        
        // decay assignments
        pe->set_value(i, d1 + d2 + d3_4 + d4_3 + d4_4);
        pae->set_value(i, d3_4 + d4_3 + d4_4);
        pm->set_value(i, d1 + d2 + d3_2 + d3_4 + d4_2 + d4_3 + d4_4);
        pam->set_value(i, d3_2 + d3_4 + d4_2 + d4_3 + d4_4);
        pt->set_value(i, d1 + d2);
        pat->set_value(i, 0);
    }
    delete energy_bins;
}

void compute_full_term(linspace_and_gl* energies_cm, double ms, double theta, double temp_cm, dummy_vars* electron, dummy_vars* anti_electron, dummy_vars* muon, dummy_vars* anti_muon, dummy_vars* tau, dummy_vars* anti_tau){
    int num_bins = energies_cm->get_len();
    dummy_vars* pe = new dummy_vars(num_bins);
    dummy_vars* pae = new dummy_vars(num_bins);
    dummy_vars* pm = new dummy_vars(num_bins);
    dummy_vars* pam = new dummy_vars(num_bins);
    dummy_vars* pt = new dummy_vars(num_bins);
    dummy_vars* pat = new dummy_vars(num_bins);

    compute_dPdtdE(energies_cm, ms, theta, temp_cm, pe, pae, pm, pam, pt, pat);
    
    for(int i = 0; i < num_bins; i++){
        double eps = energies_cm->get_value(i);
        if(eps != 0){
            double coeff = 2 * pow(_PI_,2) / pow(eps * temp_cm, 2);
            electron->set_value(i, coeff * pe->get_value(i));
            anti_electron->set_value(i, coeff * pae->get_value(i));
            muon->set_value(i, coeff * pm->get_value(i));
            anti_muon->set_value(i, coeff * pam->get_value(i));
            tau->set_value(i, coeff * pt->get_value(i));
            anti_tau->set_value(i, coeff * pat->get_value(i));
        } else {
            electron->set_value(i, 0);
            anti_electron->set_value(i, 0);
            muon->set_value(i, 0);
            anti_muon->set_value(i, 0);
            tau->set_value(i, 0);
            anti_tau->set_value(i, 0);
        }
    }

    delete pe;
    delete pae;
    delete pm;
    delete pam;
    delete pt;
    delete pat;
}
