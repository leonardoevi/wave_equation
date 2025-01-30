#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <deal.II/base/function.h>

using namespace dealii;
using namespace std;

// called f in the problem definition
template<int dim>
class ForcingTerm : public Function<dim>
{
public:
    double value(const Point<dim> & p, const unsigned int /*component*/ = 0) const override {
        const double  x = p[0], y = p[1], t = this->get_time();

        const double dtt = sin(t-x-y);
        const double dxx = sin(t-x-y);
        const double dyy = sin(t-x-y);

        return dtt - dxx - dyy;
    }
};

// called u0 in the problem definition
template<int dim>
class FunctionU0 : public Function<dim>
{
public:
    double value(const Point<dim> & p, const unsigned int /*component*/ = 0) const override {
        const double  x = p[0], y = p[1];

        return sin(x+y);
    }
};

// called u1 in the problem definition
template<int dim>
class FunctionV0 : public Function<dim>
{
public:
    double value(const Point<dim> & p, const unsigned int /*component*/ = 0) const override {
        const double  x = p[0], y = p[1];
        return -cos(-x-y);
    }
};

// called g in the problem definition
template<int dim>
class FunctionU : public Function<dim>
{
public:
    double value(const Point<dim> & p, const unsigned int /*component*/ = 0) const override {
        const double  x = p[0], y = p[1], t = this->get_time();

        return sin(x + y - t);
    }
};

// derivative of g in time
template<int dim>
class FunctiondU : public Function<dim>
{
public:
    double value(const Point<dim> & p, const unsigned int /*component*/ = 0) const override {
        const double  x = p[0], y = p[1], t = this->get_time();

        return -cos(x + y - t);
    }
};

#endif //FUNCTIONS_H
