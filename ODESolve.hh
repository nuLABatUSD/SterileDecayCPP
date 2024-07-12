#ifndef _ODESOLVE_HH_
#define _ODESOLVE_HH_
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>

#include "CashKarp_vals.hh"
#include "arrays.hh"

using std::cout;
using std::endl;
using std::ofstream;

using namespace std::chrono;

const double eps = 1e-8; 
const double TINY = 1e-40;
const double Safety = 0.9;


template <class dep>
class ODESolve
{
    protected:
        double x_value;
        dep* y_values;
        double dx_value;

    public:
        ODESolve();
        ~ODESolve();

        void set_ics(double, dep*, double);

        virtual void f(double, dep*, dep*) = 0;
        void RKCash_Karp(double, dep*, double, double*, dep*, dep*);
        bool step_accept(dep*, dep*, dep*, double, double*);
        bool RKCK_step(double, dep*, double, double*, dep*, double*);
        bool ODEOneRun(double x0, dep* y0, double dx0, int N_step, int dN, double x_final, double* x, dep* y, double* dx, const std::string& file_name, bool verbose = false);

        bool run(int N_step, int dN, double x_final, const std::string& file_name, bool verbose = false);

        void print_state();
        void print_csv(ostream&, double, double, dep*);
};

//#include "ODESolve.inl"

template <class dep>
ODESolve<dep>::ODESolve()
{
    x_value = 0.0;
    dx_value = 1.0;

}

template <class dep>
ODESolve<dep>::~ODESolve()
{
    delete y_values;
}

template <class dep>
void ODESolve<dep>::set_ics(double x0, dep* y0, double dx0)
{
    x_value = x0;
    y_values->copy(y0);
    dx_value = dx0;
}

template <class dep>
void ODESolve<dep>::RKCash_Karp(double x, dep* y, double dx, double* x_stepped, dep* y_5th, dep* y_4th)
{
    //int N;
    int N = y->length();
    //to use k1 - need to delcaire it as an array of N doubles and allocate memory (remember to delete after)
    dep* k1 = new dep(y);
    dep* k2 = new dep(y);
    dep* k3 = new dep(y);
    dep* k4 = new dep(y);
    dep* k5 = new dep(y);
    dep* k6 = new dep(y);
    
    dep* z2 = new dep(y);
    dep* z3 = new dep(y);
    dep* z4 = new dep(y);
    dep* z5 = new dep(y);
    dep* z6 = new dep(y); //inputs to get k2-k6
    
    // k1 = dx * f(x, y)
    f(x, y, k1);
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

template <class dep>
bool ODESolve<dep>::step_accept(dep* y, dep* y5, dep* y4, double dx, double* dx_new)
{
    int N = y->length();

    int problem = 0;
    
    double dsm = 0;
    double delta1 = 0;
    double delta0 = 0;

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
        //cout<< "TRUE (dsm == 0) dx_new = " << *dx_new << endl;
        return true;
    } 
    else if (dsm < 1){
        *dx_new = Safety * dx * pow(dsm, -0.2);
        *dx_new = std::min(5.0 * dx, *dx_new); 
        //cout<< "TRUE (dsm < 1) dx_new = " << *dx_new << endl;
        return true;
    }
    else{
        *dx_new = Safety * dx * pow(dsm, -0.25);
        //cout<< "FALSE dx_new = " << *dx_new << ", dsm = " << dsm << "; dx = " << dx << endl;
        //cout<< "    i= " << problem << "; y5 = " << y5->get_value(problem) << "; y4 = " << y4->get_value(problem) << endl;
        
        return false;
    }
    
    
}

template <class dep>
bool ODESolve<dep>::RKCK_step(double x, dep* y, double dx, double* x_next, dep* y_next, double* dx_next)
{
    double dx_try = dx;
    int N = y->length();
    //dep* y5(y);
    //dep* y4(y);
    dep* y5 = new dep(y); //???
    dep* y4 = new dep(y);

    bool accept = false;
    
    for (int i = 0; i<10; i++)
        
    { 
        RKCash_Karp(x, y, dx_try, x_next, y5, y4);
        if (step_accept(y, y5, y4, dx_try, dx_next))
        {
            y_next -> copy(y5);
            accept = true;
            break;
        } 
        else {
           dx_try = *dx_next; 
        }
        
    }

    if (!accept)
    {
        cout << "ERROR:  10 iterations without acceptable step" << endl;
        cout << "x = " << x << "; dx = " << dx_try << endl;
    }

    
    
    delete y5;
    delete y4;
    
    return accept;
    
    
}

template <class dep>
bool ODESolve<dep>::ODEOneRun(double x0, dep* y0, double dx0, int N_step, int dN, double x_final, double* x, dep* y, double* dx, const std::string& file_name, bool verbose) 
{
    // Set x, y, dx to initial values
    int N = y -> length();
    *x = x0;
    y -> copy(y0);
    *dx = dx0;

    // Declare for RKCK_step
    double* x_next = new double; 
    dep* y_next = new dep(y);
    double* dx_next = new double; 

    bool no_error = true;
    bool done = false;
    
    ofstream file(file_name);

    auto start = high_resolution_clock::now();

    if (verbose)
    {
        cout << "*******************" << endl;
        cout << "Running ODE Solver.  Initial Conditions:" << endl;
        print_state();
        cout << "Output printed to " << file_name << endl;
    }
    
    print_csv(file, *x, *dx, y);
    
    for (int i = 0; i < N_step && no_error && !done; i++) 
    {
        for (int j = 0; j < dN; j++) 
        {
           // cout << "ith step: " << i << ", jth step: " << j << endl;

            if (*x + *dx > x_final) 
            {
                *dx = x_final - *x;
            }
            
            if (RKCK_step(*x, y, *dx, x_next, y_next, dx_next)) 
            {
                // Update x, y, dx with the results from the RKCK step
               // cout << "Before update... x =  " << *x << "and dx = " << *dx << endl;
              
                *x = *x_next;
                y->copy(y_next);
                *dx = *dx_next;
               
               // cout << "After update... x = " << *x << "and dx = " << *dx << endl;

                
            } 
            else 
            {
                //delete x_next;
                //delete y_next;
                //delete dx_next;
                //file.close();
                no_error = false;
                break;
                //return false;
            }

            if (*x == x_final) 
            {
                cout << "Reached x_final" << endl;
                print_csv(file, *x, *dx, y);
                //delete x_next;
                //delete y_next;
                //delete dx_next;
                //file.close();
                done = true;
                break;
                //return true;
            }
        }

        print_csv(file, *x, *dx, y_next);

    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    if (verbose)
    {
        print_state();
        cout << endl << "Time elapsed: "
         << duration.count()/1000. << " seconds" << endl;

    }

    delete x_next;
    delete y_next;
    delete dx_next; 
    file.close();
    return true;
}

template <class dep>
void ODESolve<dep>::print_csv(ostream& os, double x, double dx, dep* y)
{
    os << x << ", " << dx << ", ";
    y_values->print_csv(os);
    os << endl;
}

template <class dep>
void ODESolve<dep>::print_state()
{
    cout << "x = " << x_value << ";  dx = " << dx_value << endl;
    y_values->print();
}

template <class dep>
bool ODESolve<dep>::run(int N_step, int dN, double x_final, const std::string& file_name, bool verbose)
{
    return ODEOneRun(x_value, y_values, dx_value, N_step, dN, x_final, &x_value, y_values, &dx_value, file_name, verbose);
}


#endif