#ifndef _ARRAYS_HH_
#define _ARRAYS_HH_
#include <complex>
#include <iostream>


class density;
class dep_vars;
class complex_three_vector;

using std::complex;
using std::ostream;


class dummy_vars{
    protected:
        int N;
        double* values;
        double* weights;
    public:    
        dummy_vars(int);
        dummy_vars(dummy_vars*);
        ~dummy_vars();
        void print_all();
        void set_value(int, double);
        void set_trap_weights();
        void set_weight(int, double);
        double get_value(int);
        double get_weight(int);
        int get_len();
        double integrate(dep_vars*);
};

class gl_dummy_vars : public dummy_vars
{
    public:
    gl_dummy_vars(int);
};

class gel_dummy_vars : public dummy_vars
{
    public:
    gel_dummy_vars(double, double);
};

class linspace_and_gl : public dummy_vars
{
    protected:
    int num_lin;
    int num_gl;

    public:
    linspace_and_gl(double, double, int, int);
    linspace_and_gl(linspace_and_gl*);
    double get_max_linspace();
};

class linspace_for_trap : public linspace_and_gl
{
    public:
        linspace_for_trap(double, double, int);
};


class dep_vars
{
    protected:
        int N;
        double* values;
    
    public:
        /******************************
        /  Constructors:
        /  Create an object with:
        /  (int): specify number of dependent variables, initialize them to zero
        /  (double*, int): include a pointer to an array and the length, initialize
        /  (dep_vars*): copy another dep_vars object
        *******************************/
        dep_vars(int);
        dep_vars(double*, int);
        dep_vars(dep_vars*);
        ~dep_vars();

        /*****************************
        /  Return protected values: length and individual terms values[i]
        /  Protect the array of values by returning numbers, not a pointer to the array
        *****************************/
        int length();
        double get_value(int);

        /*****************************
        /  These methods actually change the values array. Use with caution.
        /  set_value(int i, double v) sets values[i] = v
        /  copy(dep_vars*) copies the elements of another dep_vars object
        /  multiply_by(double c) takes values ==> c * values
        /  add_to(double c, dep_vars* z) takes values ==> values + c * z
        ******************************/

        void set_value(int, double);
        void zeros();
        void copy(dep_vars*);
        void multiply_by(double);
        void add_to(double, dep_vars*);

        /*****************************
        /  Methods to print to stdout
        /  print_all() prints every term on a new line (can be very long...)
        /  print(int N_top, int N_bottom) prints the first N_top terms (each on their own line) followed by "..." and then N_bottom terms.
        /  default is to print top 3, then final term... e.g., print() creates this behavior
        ******************************/

        void print_all();
        void print(int N_top = 3, int N_bot = 1);
        void print_csv(ostream& os);

            
};



class three_vector : public dep_vars
{
    public:
    /*****************************
    / Constructors:
    / three_vector() - initializes to zeros
    / three_vector(double, double, double) - uses three values to initialize x, y, z
    / three_vector(double*) - uses the first three values of the array to initialize x, y, z
    /
    / Does not need its own destructor. Just uses dep_vars destructor.
    *****************************/
    three_vector(int Nv=3);
    three_vector(double, double, double);
    three_vector(double*);
    three_vector(three_vector*);

    /******************************
    / dot_with(three_vector*), magnitude_squared(), and magnitude() return a double. They represent dot product 
    / of the vector with a second vector, itself, and the square root thereof, respectively.
    / set_cross_product(three_vector*, three_vector*) actually overwrites the components of this three_vector. Use with care.
    ******************************/

    void add(three_vector*, three_vector*);
    double dot_with(three_vector*);
    double magnitude_squared();
    double magnitude();
    void set_cross_product(three_vector*, three_vector*);
    
    void make_real(complex_three_vector*);

};

class complex_three_vector{
    protected:
    complex<double>* values;
        
    public:
    complex_three_vector(int Nv=3);
    complex_three_vector(complex<double>, complex<double>, complex<double>);
    complex_three_vector(complex<double>);
    complex_three_vector(complex_three_vector*);
    
    void print_all();
    complex<double> get_value(int);
    void set_value(int, complex<double>);
    void make_complex(three_vector*);

    void multiply_by(complex<double>);
    void add(complex_three_vector*, complex_three_vector*);
    complex<double> dot_with(complex_three_vector*);
    complex<double> magnitude_squared();
    complex<double> magnitude();
    void set_cross_product(complex_three_vector*, complex_three_vector*);
    
    ~complex_three_vector();
    
};

#endif