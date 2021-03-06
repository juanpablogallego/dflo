#ifndef __IC_H__
#define __IC_H__

//#include <cmath>

#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>

#include "claw.h"
#include "equation.h"

template <int dim>
class RayleighTaylor : public dealii::Function<dim>
{
public:
   RayleighTaylor (double gravity)
   :
   dealii::Function<dim>(MHDEquations<dim>::n_components),
   gravity (gravity)
   {}
   virtual void vector_value (const dealii::Point<dim>  &p,
                                dealii::Vector<double>  &values) const;
   
private:
   double gravity;
   const double Lx = 0.5;  // size of domain in x
   const double Ly = 1.5;  // size of domain in y
   const double A  = 0.01; // y velocity perturbation amplitude
   const double P0 = 2.5;  // pressure at y=0
};

//------------------------------------------------------------------------
// Isentropic vortex, just rotates about itself
//------------------------------------------------------------------------
template <int dim>
class IsentropicVortex : public dealii::Function<dim>
{
public:
   IsentropicVortex (double mach_inf, double theta, double beta, 
                     double x0, double y0)
   :
   dealii::Function<dim>(MHDEquations<dim>::n_components),
   mach_inf (mach_inf),
   theta (theta*M_PI/180.0),
   beta (beta),
   x0 (x0),
   y0 (y0)
   {
      const double gamma = MHDEquations<dim>::gas_gamma;
      a1 = 0.5*beta/M_PI;
      a2 = 0.5*(gamma-1.0)*std::pow(a1,2);
   }
   virtual void vector_value (const dealii::Point<dim>  &p,
                              dealii::Vector<double> &values) const; 
   
private:
   double mach_inf;
   double theta;
   double beta;
   double a1, a2;
   double x0, y0;
};

//------------------------------------------------------------------------
// Three isentropic vortices
//------------------------------------------------------------------------
template <int dim>
class VortexSystem : public dealii::Function<dim>
{
public:
   VortexSystem ()
   :
   dealii::Function<dim>(MHDEquations<dim>::n_components)
   {
      beta = 5.0;
      Rc = 4.0;
      
      const double gamma = MHDEquations<dim>::gas_gamma;
      a1 = 0.5*beta/M_PI;
      a2 = (gamma-1.0)*std::pow(a1,2)/2.0;
      
      x[0] = 0.0; y[0] = -Rc;
      x[1] = Rc*cos(30.0*M_PI/180.0); y[1] = Rc*sin(30.0*M_PI/180.0);
      x[2] = -x[1]; y[2] = y[1];
   }
   virtual void vector_value (const dealii::Point<dim>  &p,
                              dealii::Vector<double>  &values) const; 
   
private:
   double beta, Rc;
   double a1, a2;
   double x[3], y[3];
};

//------------------------------------------------------------------------
// Polarized Alfven  waves
//------------------------------------------------------------------------
template <int dim>
class AlfvenWaves : public dealii::Function<dim>
{
public:
   AlfvenWaves () : dealii::Function<dim>(MHDEquations<dim>::n_components){}
   virtual void vector_value (const dealii::Point<dim>  &p,
                              dealii::Vector<double>  &values) const;
   double value(const Point<dim> &p,
                const unsigned int component=6) const;
};

//------------------------------------------------------------------------
// Orszag-Tang vortex
//------------------------------------------------------------------------
template <int dim>
class Orszag_Tang_vortex : public dealii::Function<dim>
{
public:
   Orszag_Tang_vortex () : dealii::Function<dim>(MHDEquations<dim>::n_components){}
   virtual void vector_value (const dealii::Point<dim>  &p,
                              dealii::Vector<double>  &values) const;
};

//------------------------------------------------------------------------
// Rotor MHD
//------------------------------------------------------------------------
template <int dim>
class Rotor_MHD : public dealii::Function<dim>
{
public:
   Rotor_MHD () : dealii::Function<dim>(MHDEquations<dim>::n_components){}
   virtual void vector_value (const dealii::Point<dim>  &p,
                              dealii::Vector<double>  &values) const;
};

//------------------------------------------------------------------------
// Rotated shock tube
//------------------------------------------------------------------------
template <int dim>
class Rotated_Shock_tube : public dealii::Function<dim>
{
public:
   Rotated_Shock_tube () : dealii::Function<dim>(MHDEquations<dim>::n_components){}
   virtual void vector_value (const dealii::Point<dim>  &p,
                              dealii::Vector<double>  &values) const;
};

//------------------------------------------------------------------------
// Keplerian Disk TODO TO BE COMPLETED
//------------------------------------------------------------------------
template <int dim>
class KeplerianDisk : public dealii::Function<dim>
{
public:
   KeplerianDisk ()
   :
   dealii::Function<dim>(MHDEquations<dim>::n_components)
   {
      r0 = 0.5;
      r1 = 2.0;
      rs = 0.01;
      
      rho_out = 1.0e-6;
      rho_disk= 1.0;
      pressure= 1.0e-6;
   }
   virtual void vector_value (const dealii::Point<dim>  &p,
                              dealii::Vector<double>    &values) const;
   
private:
   double r0, r1, rs;
   double rho_out, rho_disk, pressure;
};

#endif
