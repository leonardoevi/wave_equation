#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <deal.II/base/function.h>

using namespace dealii;
using namespace std;

template<int dim>
class ForcingTerm : public Function<dim>
{
public:
    double value(const Point<dim> & p, const unsigned int /*component*/ = 0) const override {
        const double  x = p[0], t = this->get_time();

        const double dtt = sin(t-x);
        const double dxx = sin(t-x);
        //const double dyy = y*y*sin(t-sqrt(x*x+y*y))/(x*x+y*y) + x*x*cos(t-sqrt(x*x+y*y))/pow(x*x+y*y,1.5);

        return dtt - dxx;
    }
};

template<int dim>
class FunctionU0 : public Function<dim>
{
public:
    double value(const Point<dim> & p, const unsigned int /*component*/ = 0) const override {
        const double  x = p[0];

        return sin(x);
    }
};

// Function for the derivative of the initial condition.
template<int dim>
class FunctionU1 : public Function<dim>
{
public:
    double value(const Point<dim> & p, const unsigned int /*component*/ = 0) const override {
        const double  x = p[0];
        return -cos(x);
    }
};

template<int dim>
class FunctionU : public Function<dim>
{
public:
    double value(const Point<dim> & p, const unsigned int /*component*/ = 0) const override {
        const double  x = p[0], t = this->get_time();

        return sin(x - t);
    }
};

#endif //FUNCTIONS_H
