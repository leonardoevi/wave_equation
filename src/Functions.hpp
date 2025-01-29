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
        const double  x = p[0], y = p[1], t = this->get_time();

        const double dtt = sin(-t + sqrt(x*x + y*y));
        const double dxx = x*x*sin(t-sqrt(x*x+y*y))/(x*x+y*y) + y*y*cos(t-sqrt(x*x+y*y))/pow(x*x+y*y,1.5);
        const double dyy = y*y*sin(t-sqrt(x*x+y*y))/(x*x+y*y) + x*x*cos(t-sqrt(x*x+y*y))/pow(x*x+y*y,1.5);

        return dtt - dxx - dyy;
    }
};

template<int dim>
class FunctionU0 : public Function<dim>
{
public:
    double value(const Point<dim> & p, const unsigned int /*component*/ = 0) const override {
        const double  x = p[0], y = p[1];

        return sin(sqrt(x*x+y*y));
    }
};

// Function for the derivative of the initial condition.
template<int dim>
class FunctionU1 : public Function<dim>
{
public:
    double value(const Point<dim> & p, const unsigned int /*component*/ = 0) const override {
        const double  x = p[0], y = p[1];
        return -cos(sqrt(x*x+y*y));
    }
};

template<int dim>
class FunctionU : public Function<dim>
{
public:
    double value(const Point<dim> & p, const unsigned int /*component*/ = 0) const override {
        const double  x = p[0], y = p[1], t = this->get_time();

        return sin(sqrt(x*x + y*y) - t);
    }
};

#endif //FUNCTIONS_H
