#include <iostream>
#include "arrays.hh"
#include "constants.hh"
#include <cmath>
#include <complex>
#include <iomanip>
#include "gl_vals.hh"

using std::cout;
using std::endl;
using std::ostream;
using std::complex;

//dep_vars
dep_vars::dep_vars(int size)
{
    N = size;
    values = new double[N]();
}
    
dep_vars::dep_vars(double* copy_me, int size)
{
    N = size;
    values = new double[N]();
    for (int i = 0; i < N; i++)
        values[i] = copy_me[i];
}
    
dep_vars::dep_vars(dep_vars* copy_me)
{
    N = copy_me->length();
    values = new double[N]();
    for (int i = 0; i < N; i++)
        values[i] = copy_me->get_value(i);
        //values[i] = copy_me->values[i];
}
    
dep_vars::~dep_vars()
{delete[] values;}

int dep_vars::length()
{return N;}

double dep_vars::get_value(int i)
{return values[i];}

void dep_vars::set_value(int i, double v)
{values[i] = v;}

void dep_vars::zeros()
{
    for (int i = 0; i < N; i++)
        values[i] = 0.0;
}
    
void dep_vars::print_all()
{
    for (int i = 0; i < N; i++)
        cout << values[i] << endl;
}
    
void dep_vars::print(int N_top, int N_bot)
{
    if (N <= N_top + N_bot)
        print_all();
    else if (N_top < 0 || N_bot < 0)
        print_all();
    else
    {
        for (int i = 0; i < N_top; i++)
            cout << values[i] << endl;
        cout << "..." << endl;
        for (int i = 0; i < N_bot; i++)
            cout << values[N - N_bot + i] << endl;
    }
}

void dep_vars::print_csv(ostream& os)
{
    for (int i = 0; i < N-1; i++)
        os << values[i] << ", ";
    os << values[N-1];
}

void dep_vars::multiply_by(double scalar)
{
    for (int i = 0; i < N; ++i) 
    {
        values[i] *= scalar;
    }
}

void dep_vars::copy(dep_vars* z)
{
    
    for (int i = 0; i < N; ++i) 
    {
        values[i] = z -> get_value(i);
    }
}

void dep_vars::add_to(double c, dep_vars* z)
{

    for (int i = 0; i < N; ++i) {
        //values[i] += c*z[i];
        values[i] += c * z -> get_value(i);
    }

}


// three_vector
three_vector::three_vector(int Nv):dep_vars(3)
{;}

three_vector::three_vector(double x, double y, double z):dep_vars(3)
{
    values[0] = x;
    values[1] = y;
    values[2] = z;
}

three_vector::three_vector(double* copy_me):dep_vars(copy_me, 3)
{;}

three_vector::three_vector(three_vector* copy_me):dep_vars(copy_me)
{;}

void three_vector::add(three_vector* A, three_vector*B){
    values[0] = A->get_value(0) + B->get_value(0);
    values[1] = A->get_value(1) + B->get_value(1);
    values[2] = A->get_value(2) + B->get_value(2);
}

double three_vector::dot_with(three_vector* B)
{
    double dot = 0;
    for(int i = 0; i < 3; i++)
        dot += values[i] * B->get_value(i);
    return dot;
}

double three_vector::magnitude_squared()
{
    return dot_with(this);
}

double three_vector::magnitude()
{
    double sum = 0;
    for(int i =0; i < 3; i++)
        sum += pow(this->get_value(i),2);
    return sqrt(sum);
}

void three_vector::set_cross_product(three_vector* A, three_vector* B)
{
    values[0] = A->get_value(1) * B->get_value(2) - A->get_value(2) * B->get_value(1);
    values[1] = A->get_value(2) * B->get_value(0) - A->get_value(0) * B->get_value(2);
    values[2] = A->get_value(0) * B->get_value(1) - A->get_value(1) * B->get_value(0);
}

void three_vector::make_real(complex_three_vector* C)
{
    values[0] = real(C->get_value(0));
    values[1] = real(C->get_value(1));
    values[2] = real(C->get_value(2));
}

//dummy_vars
dummy_vars::dummy_vars(int num){
    N = num;
    values = new double[N]();
    weights = new double[N]();
}

dummy_vars::dummy_vars(dummy_vars* copy_me)
{
    N = copy_me->get_len();
    values = new double[N]();
    weights = new double[N]();

    for(int i = 0; i<N; i++)
        {
            values[i] = copy_me->get_value(i);
            weights[i] = copy_me->get_weight(i);
        }
}

void dummy_vars::print_all(){
    for(int i =0; i<N; i++){
        cout << values[i] << endl;
    }
}

double dummy_vars::get_value(int i){
    return values[i];
}

int dummy_vars::get_len(){
    return N;
}

double dummy_vars::get_weight(int i)
{ return weights[i]; }

void dummy_vars::set_value(int i, double v)
{values[i] = v;}

void dummy_vars::set_trap_weights(){
   weights[0] = 0.5 * (values[1] - values[0]);
   weights[N-1] = 0.5 * (values[N-1] - values[N-2]);
    for(int i=1; i<N-1; i++){
       weights[i] = 0.5 * (values[i+1] - values[i-1]);
   }
}

void dummy_vars::set_weight(int i, double w)
{weights[i] = w;}

double dummy_vars::integrate(dep_vars* fvals){
    double result = 0;
    for (int i = 0; i<N; i++){
       result += fvals->get_value(i) * weights[i]; 
    }
    return result;    
}

dummy_vars::~dummy_vars(){
    delete[] values;
    delete[] weights;
}

gl_dummy_vars::gl_dummy_vars(int num_gl):dummy_vars(num_gl)
{
    switch(num_gl){
        case 2:
            for(int i=0; i<N; i++){
                values[i] = xvals_2[i];
                weights[i] = wvals_2[i] * exp(xvals_2[i]);
             }
            break;
        case 5:
            for(int i=0; i<N; i++){
                values[i] = xvals_5[i];
                weights[i] = wvals_5[i] * exp(xvals_5[i]);
             }
            break;
        case 10:
             for(int i=0; i<N; i++){
                values[i] = xvals_10[i];
                weights[i] = wvals_10[i] * exp(xvals_10[i]);
             }
            break;
        case 50:
             for(int i=0; i<N; i++){
                values[i] = xvals_50[i];
                weights[i] = wvals_50[i] * exp(xvals_50[i]);
             }
            break;
        default:
            cout << "Error: this Gauss Laguerre number is not supported" << endl;
                
    }
}


//linspace_and_gl
linspace_and_gl::linspace_and_gl(double xmin, double xmax, int numlin, int num_gl):dummy_vars(numlin+num_gl)
{
    num_lin = numlin;
    num_gl = num_gl;
    double dx_val = (xmax - xmin) / (num_lin-1);
    for (int i = 0; i<num_lin; i++){
        values[i] = xmin + dx_val * i;
        weights[i] = dx_val;
    }
    
    weights[0] = dx_val / 2;
    weights[num_lin-1] = dx_val / 2;
    
    switch(num_gl){
        case 0:
            break;
        case 2:
            for(int i=num_lin; i<N; i++){
                values[i] = xvals_2[i-num_lin] + xmax;
                weights[i] = wvals_2[i-num_lin] * exp(xvals_2[i-num_lin]);
                
             }
            break;
        case 5:
            for(int i=num_lin; i<N; i++){
                values[i] = xvals_5[i-num_lin] + xmax;
                weights[i] = wvals_5[i-num_lin] * exp(xvals_5[i-num_lin]);
             }
            break;
        case 10:
             for(int i=num_lin; i<N; i++){
                 values[i] = xvals_10[i-num_lin] + xmax;
                weights[i] = wvals_10[i-num_lin] * exp(xvals_10[i-num_lin]);
             }
            break;
        default:
            cout << "Error: this Gauss Laguerre number is not supported" << endl;
                
    }
    
}

linspace_and_gl::linspace_and_gl(linspace_and_gl* l):dummy_vars(l->N)
{
    num_lin = l->num_lin;
    for(int i=0; i<l->N; i++){
        values[i] = l->get_value(i);
        weights[i] = l->get_weight(i);        
    }
}

double linspace_and_gl::get_max_linspace(){
    return values[num_lin-1];
    
}

//linspace_for_trap
linspace_for_trap::linspace_for_trap(double xmin, double xmax, int num):linspace_and_gl(xmin, xmax, num, 0)
{
    double dx_val = (xmax - xmin) / (N-1);
    weights[0] = dx_val / 2;
    weights[N-1] = dx_val / 2;
    for (int i=1; i<N-1; i++){
        weights[i] = dx_val;
    }
}


//complex_three_vector


complex_three_vector::complex_three_vector(int Nv){
    values = new complex<double>[3]();
    
}

complex_three_vector::complex_three_vector(complex<double> x, complex<double> y, complex<double> z){
    values = new complex<double>[3]();
    
    values[0] = x;
    values[1] = y;
    values[2] = z;
    
}

complex_three_vector::complex_three_vector(complex<double> c){
    values = new complex<double>[3]();
    for (int i = 0; i < 3; i++)
        values[i] = c;
    
    
}

complex_three_vector::complex_three_vector(complex_three_vector* c){
    values = new complex<double>[3]();
    for (int i = 0; i < 3; i++)
        values[i] = c->get_value(i);
    
}

void complex_three_vector::print_all(){
    for (int i=0; i<3; i++){
        cout << values[i] << endl;
    }
    
}

complex<double> complex_three_vector::get_value(int i){
    return values[i];
}

void complex_three_vector::set_value(complex<double> d, int i){
    values[i] = d;    
}

void complex_three_vector::add(complex_three_vector* A, complex_three_vector* B){
    values[0] = A->get_value(0) + B->get_value(0);
    values[1] = A->get_value(1) + B->get_value(1);
    values[2] = A->get_value(2) + B->get_value(2);
}

void complex_three_vector::multiply_by(complex<double> a){
   for (int i=0; i<3; i++){
       values[i] *= a;
   }
}

complex<double> complex_three_vector::dot_with(complex_three_vector* B)
{
    complex<double> dot = 0;
    for(int i = 0; i < 3; i++)
        dot += values[i] * B->get_value(i);
    return dot;
}

complex<double> complex_three_vector::magnitude_squared()
{
    return dot_with(this);
}

complex<double> complex_three_vector::magnitude()
{
    complex<double> sum = 0;
    for(int i =0; i < 3; i++)
        sum += pow(this->get_value(i),2);
    return sqrt(sum);
}

void complex_three_vector::set_cross_product(complex_three_vector* A, complex_three_vector* B)
{
    values[0] = A->get_value(1) * B->get_value(2) - A->get_value(2) * B->get_value(1);
    values[1] = A->get_value(2) * B->get_value(0) - A->get_value(0) * B->get_value(2);
    values[2] = A->get_value(0) * B->get_value(1) - A->get_value(1) * B->get_value(0);
}

void complex_three_vector::make_complex(three_vector* A){
    values[0] = complex<double> (A->get_value(0),0);
    values[1] = complex<double> (A->get_value(1),0);
    values[2] = complex<double> (A->get_value(2),0);
}


complex_three_vector::~complex_three_vector(){
   delete[] values; 
    
}