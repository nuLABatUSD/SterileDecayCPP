#include "ODESolve.hh"
#include "arrays.hh"

class expo : public ODESolve<dep_vars>
{
    public:
        expo();

        void f(double, dep_vars*, dep_vars*);

};

class spin : public ODESolve<three_vector>
{
    protected:
        three_vector* B;
        double omega;

    public:
        spin();

        void f(double, three_vector*, three_vector*);
        void print_state();
};

