#include "derivativesMPI.hh"
#include "constants.hh"
#include "decays.h"
#include "mpi.h"
#include <chrono>
#include <iostream>
#include <string>

using std::abs;
using std::cout;
using std::endl;
using std::string;
using std::to_string;

//using std::exp;

using namespace std::chrono;

//derivativesMPI::derivativesMPI(int rank, int numranks, int num, double low, double high, double start, double end, double ms, double theta, dummy_vars* freqs, double ns, double time, double temp)
derivativesMPI::derivativesMPI(int num, double low, double high, double start, double end, double ms, double theta, dummy_vars* freqs, double ns, double time, double temp)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

//    myid = rank;
//    numprocs = numranks;
    
    num_bins = num;
    E_low = low;
    E_high = high;
    a_start = start;
    a_end = end;
    sterile_mass = ms;
    mixing_angle = theta;

    y_values = new freqs_ntT(num, E_low, E_high, a_start, a_end, sterile_mass, mixing_angle, freqs, ns, time, temp);

    just_decays = new derivatives(num, low, high, start, end, ms, theta, freqs, ns, time, temp);
    
    nu_nu_tolerances = new double*[num_bins];    
    nu_nu_coll_objects = new integration*[num_bins];
    if (myid > 0)
    {
        freqs_ntT* eqm = new freqs_ntT(y_values);
        for(int j = 0; j < num_bins; j++)
        {
            double E = y_values->get_eps_value(j);
            double f = 1 / (exp(E) + 1);
            for(int k = 0; k < 6; k++)
                eqm->set_value(j + k * num_bins, f);
        }
        for(int i = myid-1; i < num_bins; i += numprocs - 1)
        {
            nu_nu_coll_objects[i] = new integration(y_values->get_eps(), i);
            nu_nu_tolerances[i] = new double[6];
            
            double dummy_ints[3][6];
            for(int j = 0; j < 2; j++)
                nu_nu_coll_objects[i]->whole_integral(eqm, a_start, j, dummy_ints[j]);
                
            for(int k = 0; k < 6; k++)
            {
                double abs_net = abs(dummy_ints[0][k] + dummy_ints[1][k]);
                double FRS = (dummy_ints[0][k] - dummy_ints[1][k]);
                if (abs_net == 0)
                    nu_nu_tolerances[i][k] = 0;
                else
                    nu_nu_tolerances[i][k] = abs_net / FRS * 10.;
            }
        }
        
        delete eqm;
        
    }
}

derivativesMPI::~derivativesMPI()
{
    delete y_values;
    delete just_decays;
    if (myid > 0)
        for(int i = myid-1; i < num_bins; i+= numprocs -1)
        {
            delete nu_nu_coll_objects[i];
            delete[] nu_nu_tolerances[i];
        }
    delete[] nu_nu_coll_objects;
    delete[] nu_nu_tolerances;
}

void derivativesMPI::set_ics(double a0, freqs_ntT* y0, double da0)
{
    x_value = a0;
    dx_value = da0;
    
    y_values->copy(y0);
}

double derivativesMPI::get_sterile_mass(){
    return sterile_mass;
}

void derivativesMPI::set_sterile_mass(double ms){
    sterile_mass = ms;
}

double derivativesMPI::get_mixing_angle(){
    return mixing_angle;
}

void derivativesMPI::set_mixing_angle(double theta){
    mixing_angle = theta;
}

double derivativesMPI::get_low(){
    return E_low;
}

double derivativesMPI::get_high(){
    return E_high;
}

void derivativesMPI::set_low(double new_low){
    E_low = new_low;
}

void derivativesMPI::set_high(double new_high){
    E_high = new_high;
}

double derivativesMPI::get_a_end(){
    return a_end;
}

void derivativesMPI::set_a_end(double end){
    a_end = end;
}

double derivativesMPI::get_a_start(){
    return a_start;
}

void derivativesMPI::set_a_start(double start){
    a_start = start;
}

void derivativesMPI::print_csv(ostream& os)
{
    os.precision(std::numeric_limits<double>::max_digits10 - 1);
    os << x_value << ", " << dx_value << ", ";
    y_values->print_csv(os);
    os << endl;
}


void derivativesMPI::f(double a, freqs_ntT* inputs, freqs_ntT* derivs)
{
    derivs->zeros();
    
   // int num_bins = inputs->get_num_bins();
    
    double* d_vals = new double[derivs->length()];
    double myans = 0;
    int sender, tag;
    
    double dummy_ints[3][6];    
    
    MPI_Status status;
    
    if(myid == 0)
    {
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
        for(int i = 0; i < num_bins; i++)
        {
            d_vals[i + 0 * num_bins] = electron->get_value(i);
            d_vals[i + 1 * num_bins] = anti_electron->get_value(i);
            d_vals[i + 2 * num_bins] = muon->get_value(i);
            d_vals[i + 3 * num_bins] = anti_muon->get_value(i);
            d_vals[i + 4 * num_bins] = tau->get_value(i);
            d_vals[i + 5 * num_bins] = anti_tau->get_value(i);

        }
        
        d_vals[0 + 6 * num_bins] = *dnsda;
        d_vals[1 + 6 * num_bins] = *dtda;
        d_vals[2 + 6 * num_bins] = *dTda;    
        
        for (int i = 0; i < num_bins; i++)
        {
            MPI_Recv(dummy_ints[2], 6, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            sender = status.MPI_SOURCE;
            tag = status.MPI_TAG;
            for(int k = 0; k < 6; k++)
                d_vals[tag + k * num_bins] += dummy_ints[2][k]; 
        }
        

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
    
    else
    {
        double net = 0;
        double FRS = 0;
    
        for(int i = myid - 1; i < num_bins; i += numprocs - 1)
        {
            for(int j = 0; j < 2; j++)
                nu_nu_coll_objects[i]->whole_integral(inputs, a, 1, dummy_ints[j]);
                
            for(int k = 0; k < 6; k++)
            {
                net = dummy_ints[0][k] + dummy_ints[1][k];
                FRS = dummy_ints[0][k] - dummy_ints[1][k];
                
                if ( abs(net) / FRS < nu_nu_tolerances[i][k])
                    dummy_ints[2][k] = 0;
                else
                    dummy_ints[2][k] = net;
                    
            }
            MPI_Send(dummy_ints[2], 6, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);
        }
    
    }
    
    MPI_Bcast(d_vals, derivs->length(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    
    for(int i = 0; i < derivs->length(); i++)
        derivs->set_value(i, d_vals[i]);
        
    delete[] d_vals;
    
}

void derivativesMPI::RKCash_Karp(double x, freqs_ntT* y, double dx_test, double* x_stepped, freqs_ntT* y_5th, freqs_ntT* y_4th)
{
    int N = y->length();
    freqs_ntT* k1 = new freqs_ntT(y);
    freqs_ntT* k2 = new freqs_ntT(y);
    freqs_ntT* k3 = new freqs_ntT(y);
    freqs_ntT* k4 = new freqs_ntT(y);
    freqs_ntT* k5 = new freqs_ntT(y);
    freqs_ntT* k6 = new freqs_ntT(y);
    
    freqs_ntT* z2 = new freqs_ntT(y);
    freqs_ntT* z3 = new freqs_ntT(y);
    freqs_ntT* z4 = new freqs_ntT(y);
    freqs_ntT* z5 = new freqs_ntT(y);
    freqs_ntT* z6 = new freqs_ntT(y); 
    
    // k1 = dx * f(x, y)
    //f(x, y, k1);
    double dx = first_derivative(x, y, k1, dx_test);
    k1 -> multiply_by(dx);  //k1 = dx * f(x,y)
  
    // k2 = dx * f(x + a2*dx, y + b21*k1)
    z2 -> copy(y);           //z2 = y
    z2 -> add_to(b21, k1);      //z2 = y + b21*k1
    f(x + a2*dx, z2, k2);          //k2 = f(x+a2*dx, z2)
    k2 -> multiply_by(dx);     //dx*f(..)

    //k2->print(8,1);
    // k3 = dx * f(x + a3*dx, y + b31*k1 + b32*k2)
    z3 -> copy(y);           //z3 = y
    z3 -> add_to(b31, k1); //z3 = y + b31*k1
    z3 -> add_to(b32, k2);
    f(x + a3*dx, z3, k3);         // k3 = f(x + a3*dx, z3)
    k3 -> multiply_by(dx);  // k3 = dx*f(x + a3*dx, z3)
 
    // k4 = dx * f(x + a4*dx, y + b41*k1 + b42*k2 +b43*k3)
    z4 -> copy(y);           //z4 = y
    z4 -> add_to(b41, k1);  //z4 = y + b41*k1
    z4 -> add_to(b42, k2); //z4 = y + b41*k1 + b42*k2
    z4 -> add_to(b43, k3); //z4 = y + b41*k1 + b42*k2 + b43*k3
    f(x + a4*dx, z4, k4);         // k4 = f(x + a4*dx, z4)
    k4 -> multiply_by(dx);
        
    // k5 = dx * f(x + a5*dx, y + b51*k1 + b52*k2 + b53*k3 + b54*k4)
    z5 -> copy(y);           //z5 = y
    z5 -> add_to(b51, k1);      //z5 = y + b51*k1
    z5 -> add_to(b52, k2);      //z5 = y + b51*k1 + b52*k2
    z5 -> add_to(b53, k3);      //z5 = y + b51*k1 + b52*k2 + b53*k3
    z5 -> add_to(b54, k4);      //z5 = y + b51*k1 + b52*k2 + b53*k3 + b54*k4    
    f(x + a5*dx, z5, k5);         // k5 = f(x + a5*dx, z5)
    k5 -> multiply_by(dx);
    
    // k6 = dx * f(x + a6*dx, y + b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5)
    z6 -> copy(y);           //z6 = y
    z6 -> add_to(b61, k1);      //z6 = y + b61*k1
    z6 -> add_to(b62, k2);      //z6 = y + b61*k1 + b62*k2
    z6 -> add_to(b63, k3);      //z6 = y + b61*k1 + b62*k2 + b63*k3
    z6 -> add_to(b64, k4);      //z6 = y + b61*k1 + b62*k2 + b63*k3 + b64*k4
    z6 -> add_to(b65, k5);      //z6 = y + b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5 
    f(x + a6*dx, z6, k6);         // k6 = f(x + a6*dx, z6)
    k6 -> multiply_by(dx);
     
    //y_5th = y + c1*k1 + c2*k2 + c3*k3 + c4*k4 + c5*k5 + c6*k6
    y_5th -> copy(y); //y_5th = y
    y_5th -> add_to(c1, k1);
    y_5th -> add_to(c2, k2);
    y_5th -> add_to(c3, k3);
    y_5th -> add_to(c4, k4);
    y_5th -> add_to(c5, k5);
    y_5th -> add_to(c6, k6);


    // y_4th = y + cstar1*k1 + cstar2*k2 + cstar3*k3 + cstar4*k4 + cstar5*k5 + cstar6*k6
    y_4th -> copy(y); //y_4th = y           
    y_4th -> add_to(cstar1, k1); //y_4th = y + cstar1*k1
    y_4th -> add_to(cstar2, k2); //y_4th = y + cstar1*k1 + cstar2*k2
    y_4th -> add_to(cstar3, k3); //y_4th = y + cstar1*k1 + cstar2*k2 + cstar3*k3
    y_4th -> add_to(cstar4, k4); //y_4th = y + cstar1*k1 + cstar2*k2 + cstar3*k3 + cstar4*k4
    y_4th -> add_to(cstar5, k5); //y_4th = y + cstar1*k1 + cstar2*k2 + cstar3*k3 + cstar4*k4 + cstar5*k5
    y_4th -> add_to(cstar6, k6); //y_4th = y + cstar1*k1 + cstar2*k2 + cstar3*k3 + cstar4*k4 + cstar5*k5 + cstar6*k6

    // x_stepped = x + dx
    *x_stepped = x + dx;
    delete k1;
    delete k2;
    delete k3;
    delete k4;
    delete k5;
    delete k6;
    
    delete z2;
    delete z3;
    delete z4;
    delete z5;
    delete z6;
    return;

}

bool derivativesMPI::step_accept(freqs_ntT* y, freqs_ntT* y5, freqs_ntT* y4, double dx, double* dx_new, bool error_verbose, bool print_error_file)
{
    int N = y->length();

    int problem = 0;
    
    double dsm = 0;
    double delta1 = 0;
    double delta0 = 0;
    
    bool accept;

    if(myid == 0)
    {
        for (int i = 0; i<N; i++)
        { 
            delta1 = abs(y5 -> get_value(i) - y4 -> get_value(i));
            delta0 = eps*(abs(y -> get_value(i)) + abs(y5 -> get_value(i) - y -> get_value(i))) + TINY;
            
    
            if (delta1/delta0 > dsm)
            { 
                dsm = delta1/delta0;
                problem = i;
                
             }
         }
         
        if (dsm == 0)
        {
            *dx_new = 5 * dx;
            
            accept = true;
        } 
        else if (dsm < 1){
            *dx_new = Safety * dx * pow(dsm, -0.2);
            *dx_new = std::min(5.0 * dx, *dx_new); 
            accept = true;
        }
        else{
            *dx_new = Safety * dx * pow(dsm, -0.25);
            *dx_new = std::min(0.5 * dx, *dx_new);
            if (error_verbose)
                cout << "dsm = " << dsm << ", dx = " << dx << endl << "problem index = " << problem << "; y5 = " << y5->get_value(problem) << "; y4 = " << y4->get_value(problem) << endl;
                
            if (print_error_file)
            {
                ofstream error_file("ODESolve_ERROR_STEP_ACCEPT.csv");
                print_csv(error_file);
                error_file.close();
                
                ofstream deriv_error_file("ODESolve_ERROR_fproblem.csv");
                
                deriv_error_file.precision(std::numeric_limits<double>::max_digits10 - 1);
                double x_temp = x_value;
                freqs_ntT* y_temp = new freqs_ntT(y_values);
                freqs_ntT* f_temp = new freqs_ntT(y_values);
                int N_values = 101;
                double dx_temp = dx_value / (N_values-1);
                for (int i = 0; i<N_values; i++)
                {
                    f(x_temp, y_temp, f_temp);
                    y_temp->add_to(dx_temp, f_temp);
                    deriv_error_file << x_temp << ", " << f_temp->get_value(problem) << ", " << y_temp->get_value(problem) << ", " << f_temp->get_value(problem-1) << ", " << f_temp->get_value(problem+1) << endl;
                    x_temp += dx_temp;
                }
                deriv_error_file.close();
                
                delete y_temp;
                delete f_temp;
                
            }
            total_ODE_rejected_steps++;
            accept = false;
        }
    }
    
    MPI_Bcast(&accept, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    return accept;
}

bool derivativesMPI::RKCK_step(double x, freqs_ntT* y, double dx, double* x_next, freqs_ntT* y_next, double* dx_next)
{
    double dx_try = dx;
    double dx_future, x_future;
    int N = y->length();
    freqs_ntT* y5 = new freqs_ntT(y); 
    freqs_ntT* y4 = new freqs_ntT(y);
    bool accept = false;
    
    for (int i = 0; i<10; i++)        
    { 
        RKCash_Karp(x, y, dx_try, &x_future, y5, y4);
        if (step_accept(y, y5, y4, dx_try, &dx_future))
        {
            y_next -> copy(y5);
            *dx_next = dx_future;
            *x_next = x_future;
            accept = true;
            break;
        } 
        else {
            if (i < 10)
               dx_try = dx_future; 
        }
        
    }

    if(myid == 0)
    {
        if (!accept)
        {
            cout << "ERROR:  10 iterations without acceptable step" << endl;
            cout << "x = " << x << "; dx = " << dx_try << endl;
            
            dx_try = dx;
            for (int i =0; i < 10; i++)
            {
                cout << "Step " << i << " ";
                RKCash_Karp(x, y, dx_try, &x_future, y5, y4);
                if (i < 9)  
                    step_accept(y, y5, y4, dx_try, &dx_future, true, false);
                else
                    step_accept(y, y5, y4, dx_try, &dx_future, true, true);
                dx_try = dx_future;
            }
        }
    }    
    delete y5;
    delete y4;
    
    return accept;

}

bool derivativesMPI::RKCK_step_advance()
{   return RKCK_step(x_value, y_values, dx_value, &x_value, y_values, &dx_value);  }

bool derivativesMPI::ODEOneRun(int N_step, int dN, double x_final, const std::string& file_name, bool verbose, bool print_csv_file)
{
    int N = y_values->length();
    
    // Initial values are (x_value, y_values, dx_value) as set by set_ics
    
    // Declare for RKCK_step
    
    bool no_error= true;
    bool done = false;
    
    if (x_final <= x_value)
    {
        if(myid == 0)
            cout << "x_final = " << x_final << " is less than initial condition, x_value = " << x_value << endl;
        return true;
    }
    
    if(myid == 0)
    {
        ofstream file;
        if (print_csv_file)
            file.open(file_name);
        
        auto start = high_resolution_clock::now();
        
        if (verbose)
        {
            cout << "*******************" << endl;
            cout << "Running ODE Solver." << endl;
            cout << "Output printed to " << file_name << endl;
        }
    
        if (print_csv_file)
            print_csv(file);    
        
        for (int i = 0; i < N_step && no_error && !done; i++)
        {
            for(int j = 0; j < dN; j++)
            {
                cout << i << j << endl;
                if(x_value + dx_value > x_final)
                    dx_value = x_final - x_value;
                
                if (!RKCK_step_advance())
                {
                    no_error = false;
                    break;
                }
                total_ODE_steps++;
                
                if (x_value == x_final)
                {
                    if(verbose)
                        cout << "Reached x_final" << endl;
                    if (print_csv_file)
                        print_csv(file);
                    done = true;
                    break;
                }
            }
            if (!done && print_csv_file)
                print_csv(file);
        }
        
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
    
        if (verbose)
        {
            cout << endl << "Time elapsed: "
             << duration.count()/1000. << " seconds" << endl;
            cout << "steps rejected / total steps = " << total_ODE_rejected_steps << " / " << total_ODE_steps << " (" << 100-(100 * total_ODE_rejected_steps) / (total_ODE_rejected_steps + total_ODE_steps) << "% effciency)" << endl;
    
        }
    
        if(print_csv_file)
            file.close();
    }
    else
    {
        for (int i = 0; i < N_step && no_error && !done; i++)
        {
            for(int j = 0; j < dN; j++)
            {
                if(x_value + dx_value > x_final)
                    dx_value = x_final - x_value;
                
                if (!RKCK_step_advance())
                {
                    no_error = false;
                    break;
                }
                total_ODE_steps++;
                
                if (x_value == x_final)
                {
                    done = true;
                    break;
                }
            }
        }
            
    }
    return done;
}

double derivativesMPI::first_derivative(double a, freqs_ntT* inputs, freqs_ntT* derivs, double dx_test)
{
    derivs->zeros();
    
    double* d_vals = new double[inputs->length()];
    double myans = 0;
    int sender, tag;
    
    double dummy_ints[3][6];    
    
    MPI_Status status;
    
    double new_dx = 0;
    
    if(myid == 0)
    {
        double* next_dx = new double;
        double* x_next = new double;
        freqs_ntT* y_next = new freqs_ntT(inputs);
        
        just_decays->RKCK_step(a, inputs, dx_test, x_next, y_next, next_dx);
        
        new_dx = *x_next - a;
        
        delete next_dx;
        delete x_next;
        delete y_next;
        
        
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
        for(int i = 0; i < num_bins; i++)
        {
            d_vals[i + 0 * num_bins] = electron->get_value(i);
            d_vals[i + 1 * num_bins] = anti_electron->get_value(i);
            d_vals[i + 2 * num_bins] = muon->get_value(i);
            d_vals[i + 3 * num_bins] = anti_muon->get_value(i);
            d_vals[i + 4 * num_bins] = tau->get_value(i);
            d_vals[i + 5 * num_bins] = anti_tau->get_value(i);

        }
        
        d_vals[0 + 6 * num_bins] = *dnsda;
        d_vals[1 + 6 * num_bins] = *dtda;
        d_vals[2 + 6 * num_bins] = *dTda;    
        
        for (int i = 0; i < num_bins; i++)
        {
            MPI_Recv(dummy_ints[2], 6, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            sender = status.MPI_SOURCE;
            tag = status.MPI_TAG;
            for(int k = 0; k < 6; k++)
                d_vals[tag + k * num_bins] += dummy_ints[2][k]; 
        }
        

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
    
    else
    {
        double net = 0;
        double FRS = 0;
    
        for(int i = myid - 1; i < num_bins; i += numprocs - 1)
        {
            for(int j = 0; j < 2; j++)
                nu_nu_coll_objects[i]->whole_integral(inputs, a, 1, dummy_ints[j]);
                
            for(int k = 0; k < 6; k++)
            {
                net = dummy_ints[0][k] + dummy_ints[1][k];
                FRS = dummy_ints[0][k] - dummy_ints[1][k];
                
                if ( abs(net) / FRS < nu_nu_tolerances[i][k])
                    dummy_ints[2][k] = 0;
                else
                    dummy_ints[2][k] = net;
                    
            }
            MPI_Send(dummy_ints[2], 6, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);
        }
    
    }
    
    MPI_Bcast(d_vals, derivs->length(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    
    for(int i = 0; i < derivs->length(); i++)
        derivs->set_value(i, d_vals[i]);
        
    delete[] d_vals;
    
    MPI_Bcast(&new_dx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
//    if (myid == 0)
//        cout << "First derivative rejected " << just_decays->get_rejected_steps() << " steps from decays alone" << endl;
    
    return new_dx;

}

dummy_vars* derivativesMPI::retrieve_separations(){
    return y_values->get_separations();
}

void derivativesMPI::update(double new_a_start, double new_a_end){
    if (a_start != new_a_start || a_end != new_a_end)
    {
        a_start = new_a_start;
        a_end = new_a_end;
        y_values->eps_shift(new_a_start, new_a_end);
    }
    
    if (myid > 0)
    {
        freqs_ntT* eqm = new freqs_ntT(y_values);
        double dummy_ints[3][6];
        
        for(int j = 0; j < num_bins; j++)
        {
            double E = y_values->get_eps_value(j);
            double f = 1 / (exp(E) + 1);
            for(int k = 0; k < 6; k++)
                eqm->set_value(j + k * num_bins, f);
        }
        for(int i = myid-1; i < num_bins; i += numprocs - 1)
        {
            delete nu_nu_coll_objects[i];
            nu_nu_coll_objects[i] = new integration(y_values->get_eps(), i);
            
            for(int j = 0; j < 2; j++)
                nu_nu_coll_objects[i]->whole_integral(eqm, a_start, j, dummy_ints[j]);
                
            for(int k = 0; k < 6; k++)
            {
                double abs_net = abs(dummy_ints[0][k] + dummy_ints[1][k]);
                double FRS = (dummy_ints[0][k] - dummy_ints[1][k]);
                if (FRS==0)
                    cout << "**" << i << abs_net << y_values->get_eps_value(i) << endl;
                if (abs_net == 0)
                    nu_nu_tolerances[i][k] = 0;
                else
                    nu_nu_tolerances[i][k] = abs_net / FRS * 10.;
            }
            cout << i << eqm->get_value(i) << "***" << nu_nu_tolerances[i][0] << ", " << y_values->get_eps_value(i) << endl;
        }
             
         delete eqm;   
    }

    
}

void derivativesMPI::print_eps_file(const std::string& file_name)
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

bool derivativesMPI::inch_forward(double a0, double a1, string& folder_name, string& file_name, bool verbose)
{
    if (x_value != a0)
    {
        cout << "ERROR: The initial a must be equal to a_low" << endl;
        return false;
    }
    
    update(a0, a1);
    dummy_vars* a_separations = retrieve_separations();
    dx_value *= 0.01;
    
    
    string data_file_name = folder_name + "/data-" + file_name + ".csv";
    string eps_file_name = folder_name + "/eps-" + file_name + ".csv";
    
    if(myid == 0)
    {
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
    }
    else
    {
        for(int i = 0; i < a_separations->get_len(); i++)
            if(!ODEOneRun(1000, 1, a_separations->get_value(i), "no_file", false, false))
                return false;
    }
    delete a_separations;
    
    
    return true;
}
