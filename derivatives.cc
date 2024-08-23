#include "ODESolve.hh"
#include "arrays.hh"
#include "freqs_ntT.hh"
#include "derivatives.hh"
#include "constants.hh"
#include "decays.h"
#include "gl_vals.hh"
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>

using std::cout;
using std::endl;
using std::to_string;
using std::string;
using std::ofstream;

void get_full_term(freqs_ntT*, double**, double**, double, int, bool);
void compute_equilibrium(gel_linspace_gl*, double***, double);

derivatives::derivatives(int num, double low, double high, double start, double end, double ms, double theta, dummy_vars* freqs, double ns, double time, double temp):ODESolve(){
    num_bins = num;
    E_low = low;
    E_high = high;
    a_start = start;
    a_end = end;
    sterile_mass = ms;
    mixing_angle = theta;

    y_values = new freqs_ntT(num, E_low, E_high, a_start, a_end, sterile_mass, mixing_angle, freqs, ns, time, temp);

    Rvi = new double**[3];
    for(int i = 0; i < 3; i++){
        Rvi[i] = new double*[6];
        for(int j = 0; j < 6; j++){
            Rvi[i][j] = new double[num_bins];
        }
    }

    compute_equilibrium(y_values->get_eps(), Rvi, 1);
}

derivatives::~derivatives(){
    delete y_values;
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 6; j++){
            delete[] Rvi[i][j];
        }
        delete[] Rvi[i];
    }
    delete[] Rvi;
}

dummy_vars* derivatives::retrieve_separations(){
    return y_values->get_separations();
}

freqs_ntT* derivatives::get_yvalues(){
    return y_values;
}

void derivatives::update(double new_a_start, double new_a_end){
    y_values->eps_shift(new_a_start, new_a_end);
    compute_equilibrium(y_values->get_eps(), Rvi, 1);
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

void get_full_term(freqs_ntT* input, double** forward, double** backward, double a, int p1, bool equilibrium){
    gel_linspace_gl* e = new gel_linspace_gl(input->get_eps());
    integration* col_vv = new integration(e, p1);
    nu_e_collision_R1* col_R1 = new nu_e_collision_R1(e, p1, a, equilibrium);
    nu_e_collision_R2* col_R2 = new nu_e_collision_R2(e, p1, a, equilibrium);
    
    col_vv->whole_integral(input, a, 0, forward[0]);
    col_vv->whole_integral(input, a, 1, backward[0]);

    col_R1->whole_integral(input, 0, forward[1]);
    col_R1->whole_integral(input, 1, backward[1]);

    col_R2->whole_integral(input, 0, forward[2]);
    col_R2->whole_integral(input, 1, backward[2]);

    delete col_vv;
    delete col_R1;
    delete col_R2;
    delete e;
}

/*
void is_zero(freqs_ntT* input, gel_linspace_gl* e, double* forward, double* backward, double a, int p1, double threshold){
    double* forward_result = new double[6];
    double* backward_result = new double[6];
    double* forward_append = new double[6];
    double* backward_append = new double[6];


    integration* col_vv = new integration(e, p1);
    col_vv->whole_integral_eq(a, 0, forward_result);
    col_vv->whole_integral_eq(a, 1, backward_result);
    
    nu_e_collision_R1* col_R1 = new nu_e_collision_R1(e, p1, a, );
    col_R1->whole_integral_eq(input, 0, forward_append);
    col_R1->whole_integral_eq(input, 1, backward_append);
    for(int i = 0; i < 6; i++){
        forward_result[i] += forward_append[i];
        backward_result[i] += backward_append[i];
    }

    nu_e_collision_R2* col_R2 = new nu_e_collision_R2(e, p1, a);
    col_R2->whole_integral_eq(input, 0, forward_append);
    col_R2->whole_integral_eq(input, 1, backward_append);
    for(int i = 0; i < 6; i++){
        forward_result[i] += forward_append[i];
        backward_result[i] += backward_append[i];
    }

    double R;
    double ratio;
    for(int i = 0; i < 6; i++){
        R = fabs(forward_result[i] - backward_result[i]) / (forward_result[i] + backward_result[i]);
        ratio = fabs(forward[i] - backward[i]) / (forward[i] + backward[i]);
        cout << "Neutrino " << i << ": " << ratio / R << ", R = " << R << ", ratio = " << ratio << endl;
    }
    delete[] forward_result;
    delete[] forward_append;
    delete[] backward_append;
    delete[] backward_result;

    delete col_vv;
    delete col_R1;
    delete col_R2;
}*/

void compute_equilibrium(gel_linspace_gl* eps, double*** Rvi, double a){
    freqs_ntT* use = new freqs_ntT(eps);
    double** forward = new double*[3];
    double** backward = new double*[3];
    for(int j = 0; j < 3; j++){
        forward[j] = new double[6];
        backward[j] = new double[6];
    }
    for(int i = 0; i < eps->get_len(); i++){
        get_full_term(use, forward, backward, a, i, true);
        for(int j = 0; j < 3; j++){
            for(int k = 0; k < 6; k++){
                Rvi[j][k][i] = fabs(forward[j][k] - backward[j][k]) / (forward[j][k] + backward[j][k]);
            }
        }
    }

    for(int i = 0; i < 3; i++){
        delete[] forward[i];
        delete[] backward[i];
    }
    delete[] forward;
    delete[] backward;
    delete use;
}

void compute_scaled_errors(gel_linspace_gl* eps, double*** forward, double*** backward, double* dn, double* dp){
    double topn = 0;
    double botn = 0;
    double topp = 0;
    double botp = 0;

    dep_vars* integrator_n1 = new dep_vars(eps->get_len());
    dep_vars* integrator_n2 = new dep_vars(eps->get_len());
    dep_vars* integrator_p1 = new dep_vars(eps->get_len());
    dep_vars* integrator_p2 = new dep_vars(eps->get_len());

    for(int i = 0; i < 6; i++){

        for(int j = 0; j < eps->get_len(); j++){
            double for_app = 0;
            double bac_app = 0;

            for(int k = 0; k < 3; k++){
                for_app += forward[j][k][i];
                bac_app += backward[j][k][i];
            }

            integrator_n1->set_value(j,pow(eps->get_value(j), 2) * (for_app - bac_app));
            integrator_n2->set_value(j,pow(eps->get_value(j), 2) * (for_app + bac_app));
            integrator_p1->set_value(j,pow(eps->get_value(j), 3) * (for_app - bac_app));
            integrator_p2->set_value(j,pow(eps->get_value(j), 3) * (for_app + bac_app));
        }

        topn += eps->integrate(integrator_n1);
        botn += eps->integrate(integrator_n2);
        topp += eps->integrate(integrator_p1);
        botp += eps->integrate(integrator_p2);
    }

    *dn = topn / botn;
    *dp = topp / botp;

    delete integrator_n1;
    delete integrator_n2;
    delete integrator_p1;
    delete integrator_p2;
}

void derivatives::f(double a, freqs_ntT* inputs, freqs_ntT* derivs){
    int num_bins = inputs->get_num_bins();
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

    double* collision_terms = new double[6];
    
    double*** forward = new double**[num_bins];
    double*** backward = new double**[num_bins];
    for(int i = 0; i < num_bins; i++){
        forward[i] = new double*[3];
        backward[i] = new double*[3];
        for(int j = 0; j < 3; j++){
            forward[i][j] = new double[6];
            backward[i][j] = new double[6];
        }
    }
    
    for(int i = 0; i < num_bins; i++){

        for(int j = 0; j < 6; j++){
            collision_terms[j] = 0;
        }

        
        get_full_term(inputs, forward[i], backward[i], a, i, false);

        double ratio = 0;
        for(int k = 0; k < 3; k++){
            for(int j = 0; j < 6; j++){
                ratio = fabs(forward[i][k][j] - backward[i][k][j]) / (forward[i][k][j] + backward[i][k][j]);
                if((ratio / Rvi[k][j][i]) > _tolerance_){
                    collision_terms[j] += (forward[i][k][j] - backward[i][k][j]) * *dtda;
                } 
            }
        }
        
        
        derivs->set_value(i, electron->get_value(i) + collision_terms[0]);
        derivs->set_value(i + num_bins, anti_electron->get_value(i) + collision_terms[1]);
        derivs->set_value(i + 2 * num_bins, muon->get_value(i) + collision_terms[2]);
        derivs->set_value(i + 3 * num_bins, anti_muon->get_value(i) + collision_terms[3]);
        derivs->set_value(i + 4 * num_bins, tau->get_value(i) + collision_terms[4]);
        derivs->set_value(i + 5 * num_bins, anti_tau->get_value(i) + collision_terms[5]);
    }
    //file.precision(std::numeric_limits<double>::max_digits10);
    //file << a << ", " << derivs->get_value(2) << endl;
    derivs->set_ns(*dnsda);
    derivs->set_time(*dtda);
    derivs->set_temp(*dTda);
    
    double* dn = new double;
    double* dp = new double;

    compute_scaled_errors(inputs->get_eps(), forward, backward, dn, dp);
    cout << "BEHOLD: " << *dn << ", " << *dp << endl;

    delete dn;
    delete dp;
    for(int i = 0; i < num_bins; i++){
        for(int j = 0; j < 3; j++){
            delete[] forward[i][j];
            delete[] backward[i][j];
        }
        delete[] forward[i];
        delete[] backward[i];
    }
    delete[] forward;
    delete[] backward;
    delete[] collision_terms;
    

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
    * must obey the inequality scales_num >= floor[log( a_end / a_start ) / log(E_low * (num - 16) / E_high) + 2]
    */
    double a_start = 0.1;
    double a_end = 0.25;
    int num = 100;
    int scale_num = 3;
    
    double ms = 245;
    double theta = 1.22e-5;
    double time = 0;
    double temp = 1 / a_start;

    /* for collision tests*/
    double a = 0.1;
    temp = 1/a;
    
    double E_low = min_low(ms);
    double E_high = (ms) / 2;
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

    //Acceptance Tolerance Testing
    /*int p1_idx = 17;
    double*** Rvi = new double**[3];
    for(int i = 0; i < 3; i++){
        Rvi[i] = new double*[3];
        for(int j = 0; j < 3; j++){
            Rvi[i][j] = new double[eps->get_len()];
        }
    }
    compute_equilibrium(eps, Rvi, a);
    double** forward = new double*[3];
    double** backward = new double*[3];
    for(int j = 0; j < 3; j++){
        forward[j] = new double[6];
        backward[j] = new double[6];
    }
    freqs_ntT* use = new freqs_ntT(eps);
    get_full_term(input, forward, backward, a, 8, false);
    for(int i = 0 ; i < 3; i++){
        cout << Rvi[i][0][8] << ", " << fabs(forward[i][0] - backward[i][0]) / (forward[i][0] + backward[i][0]) << endl;
    }*/

    /*nu_e_collision_R2* col = new nu_e_collision_R2(eps, p1_idx, a);
    nu_e_collision_R1* col2 = new nu_e_collision_R1(eps, p1_idx, a);
    double p1_energy = eps->get_value(p1_idx);
    cout<<"p1 = " << p1_energy << endl;
    double* results = new double[6];
    double* results2 = new double[6];
    cout << "R1 Terms" << endl;
    col2->populate_F(input, 0);
    col2->whole_integral(input, 0, results);
    col2->populate_F(input, 1);
    col2->whole_integral(input, 1, results2);
    for(int i = 0; i < 6; i++){
        cout << "Forward: " << results[i] << ", Backward: " << results2[i] << ", percentage diff = " << 200 * (results[i] - results2[i]) / (results[i] + results2[i]) << "%" << endl;
    }
    double* results3 = new double[6];
    double* results4 = new double[6];
    cout << "R2 Terms" << endl;
    col->populate_F(input, 0);
    col->whole_integral(input, 0, results3);
    col->populate_F(input, 1);
    col->whole_integral(input, 1, results4);
    for(int i = 0; i < 6; i++){
        cout << "Forward: " << results3[i] << ", Backward: " << results4[i] << ", percentage diff = " << 200 * (results3[i] - results4[i]) / (results3[i] + results4[i]) << "%" << endl;
    }

    integration* inter = new integration(eps, p1_idx);
    double* results5 = new double[6];
    double* results6 = new double[6];
    inter->whole_integral(input, a, 0, results5);
    inter->whole_integral(input, a, 1, results6);
    for(int i = 0; i < 6; i++){
        double one = results5[i];
        double two = results6[i];
        cout <<  one << ", " << two << ", " << one - two << ", " << 200 * (one - two) / (one + two) << "%" << endl;
    }
    cout << "here" << endl;
    nu_e_collision_R2* intern = new nu_e_collision_R2(eps, p1_idx, a);
    double* results7 = new double[6];
    double* results8 = new double[6];
    intern->whole_integral_eq(input, 0, results7);
    intern->whole_integral_eq(input, 1, results8);
    for(int i = 0; i < 6; i++){
        double one = results7[i];
        double two = results8[i];
        cout <<  one << ", " << two << ", " << one - two << ", " << 200 * (one - two) / (one + two) << "%" << endl;
    }*/


    // testing code
    /*int p1_idx = 17;
    nu_e_collision_R2* col = new nu_e_collision_R2(eps, p1_idx, a);
    nu_e_collision_R1* col2 = new nu_e_collision_R1(eps, p1_idx, a);
    double p1_energy = eps->get_value(p1_idx);
    cout<<"p1 = " << p1_energy << endl;
    double* results = new double[6];
    double* results2 = new double[6];
    cout << "R1 Terms" << endl;
    col2->populate_F(input, 0);
    col2->whole_integral(input, 0, results);
    col2->populate_F(input, 1);
    col2->whole_integral(input, 1, results2);
    for(int i = 0; i < 6; i++){
        cout << "Forward: " << results[i] << ", Backward: " << results2[i] << ", percentage diff = " << 200 * (results[i] - results2[i]) / (results[i] + results2[i]) << "%" << endl;
    }
    double* results3 = new double[6];
    double* results4 = new double[6];
    cout << "R2 Terms" << endl;
    col->populate_F(input, 0);
    col->whole_integral(input, a, 0, results3);
    col->populate_F(input, 1);
    col->whole_integral(input, a, 1, results4);
    for(int i = 0; i < 6; i++){
        cout << "Forward: " << results3[i] << ", Backward: " << results4[i] << ", percentage diff = " << 200 * (results3[i] - results4[i]) / (results3[i] + results4[i]) << "%" << endl;
    }

    integration* inter = new integration(eps, p1_idx);
    double* results5 = new double[6];
    double* results6 = new double[6];
    inter->whole_integral(input, 0.1, 0, results5);
    inter->whole_integral(input, 0.1, 1, results6);
    for(int i = 0; i < 6; i++){
        double one = results5[i];
        double two = results6[i];
        cout <<  one << ", " << two << ", " << one - two << ", " << 200 * (one - two) / (one + two) << "%" << endl;
    }
    delete inter;
    delete results;
    delete results2;*/

    sim->set_ics(a_start, input, 0.01 * a_start);
    dummy_vars* a_separations = sim->retrieve_separations();
    a_separations->print_all();
    for(int i = 0; i < a_separations->get_len(); i++){
        double a = a_separations->get_value(i);
        cout << a << endl;
        string name = "oua" + to_string(file_idx) + ".csv";
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
        freqs_ntT* use = sim->get_yvalues();
        gel_linspace_gl* use2 = use->get_eps();
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