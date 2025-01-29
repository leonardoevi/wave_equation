#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <deal.II/base/function.h>

using namespace dealii;

template<int dim>
class FunctionMu : public Function<dim>
{
public:
     double value(const Point<dim> & /*p*/, const unsigned int /*component*/ = 0) const override {
        return 1.0;
    }
};

template<int dim>
class ForcingTerm : public Function<dim>
{
public:
    double value(const Point<dim> & p, const unsigned int /*component*/ = 0) const override {
        double x = p[0], y = p[1];
        if ((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5) < 0.001)
            return 6*std::sin(8*M_PI*this->get_time());

        return 0.0;
    }
};

template<int dim>
class FunctionU0 : public Function<dim>
{
public:
    double value(const Point<dim> & /*p*/, const unsigned int /*component*/ = 0) const override {
        return 0.0;
        /*
        double x = p[0], y = p[1];
        if (x <= 0.5)
          return std::sin(6.283185 * x) * std::sin(6.283185 * x) * std::sin(6.283185 / 2.0 * y) * std::sin(6.283185 / 2.0 * y);
        else
          return 0.0;
          */
        //return std::sin(3.14 * 2 * p[0]) * std::sin(3.14 * 2 * p[1]);
        //return exp(-30*(p[0] - 0.5)*(p[0] - 0.5) - 30 * (p[1] - 0.5)*(p[1] - 0.5));
    }
};

// Function for the derivative of the initial condition.
template<int dim>
class FunctionU1 : public Function<dim>
{
public:
    double value(const Point<dim> & /*p*/, const unsigned int /*component*/ = 0) const override {
        return 0.0;
    }
};

#endif //FUNCTIONS_H
