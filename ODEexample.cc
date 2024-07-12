#include <iostream>
#include "constants.hh"
#include "ODEexample.hh"

using std::cout;
using std::endl;

expo::expo() : ODESolve()
{
    y_values = new dep_vars(4);
}

void expo::f(double x, dep_vars* y, dep_vars* z)
{
    z->set_value(0, y->get_value(1));
    z->set_value(1, y->get_value(2));
    z->set_value(2, y->get_value(3));
    z->set_value(3, pow(_PI_, 4) * y->get_value(0));
    return;
}

spin::spin() : ODESolve()
{
    B = new three_vector(0.8, 0.0, 0.6);
    omega = _PI_;
    y_values = new three_vector(3);
}

void spin::f(double x, three_vector* y, three_vector* z)
{
    z->set_cross_product(B, y);
    z->multiply_by(omega);
}

void spin::print_state()
{
    ODESolve::print_state();
    cout << "| L | = " << y_values->magnitude() << endl;
}

int main()
{    
    expo* sim = new expo;
    double ics[] = {1.0, 0.0, - _PI_ * _PI_, 0.0};

    dep_vars* y0 = new dep_vars(ics, 4);
    sim->set_ics(0, y0, 0.1);

    sim->print_state();

    sim->run(100,10,5.0,"output2.csv");

    sim->print_state();

    cout << "======================" << endl;

    spin* sim2 = new spin;
    three_vector* L0 = new three_vector(0., 0., 1.);

    sim2->set_ics(0, L0, 0.1);
    sim2->print_state();

    sim2->run(100, 10, 5.0, "output3.csv");
    sim2->print_state();
    return 1;
}