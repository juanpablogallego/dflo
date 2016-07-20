#ifndef __EQUATION_H__
#define __EQUATION_H__

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/lac/parallel_vector.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>

#include<cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

//------------------------------------------------------------------------------
inline
double logavg(double a, double b)
{
   double xi = b/a;
   double f = (xi - 1.0) / (xi + 1.0);
   double u = f * f;
   
   double F;
   if (u < 1.0e-2)
   {
      double u2 = u * u;
      double u3 = u2 * u;
      F = 1.0 + u/3.0 + u2/5.0 + u3/7.0;
   }
   else
      F = log(xi)/2.0/f;
   
   return 0.5*(a+b)/F;
}

template <int dim>
struct EulerEquations
{
   // First dim components correspond to momentum
   static const unsigned int n_components             = dim + 2;
   static const unsigned int density_component        = dim;
   static const unsigned int energy_component         = dim+1;
   
   static
   std::vector<std::string>
   component_names ()
   {
      std::vector<std::string> names;
      names.push_back ("XMomentum");
      names.push_back ("YMomentum");
      if(dim==3)
         names.push_back ("ZMomentum");
      names.push_back ("Density");
      names.push_back ("Energy");
      
      return names;
   }
   
   
   static
   std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
   component_interpretation ()
   {
      std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation
      (dim, dealii::DataComponentInterpretation::component_is_part_of_vector);
      data_component_interpretation
      .push_back (dealii::DataComponentInterpretation::component_is_scalar);
      data_component_interpretation
      .push_back (dealii::DataComponentInterpretation::component_is_scalar);
      
      return data_component_interpretation;
   }
   
   // Ratio of specific heats
   static const double gas_gamma;
   
   //---------------------------------------------------------------------------
   // Compute kinetic energy from conserved variables
   //---------------------------------------------------------------------------
   template <typename number, typename InputVector>
   static
   number
   compute_kinetic_energy (const InputVector &W)
   {
      number kinetic_energy = 0;
      for (unsigned int d=0; d<dim; ++d)
         kinetic_energy += *(W.begin()+d) *
                           *(W.begin()+d);
      kinetic_energy *= 0.5/(*(W.begin() + density_component));
      
      return kinetic_energy;
   }
   
   //---------------------------------------------------------------------------
   // Compute pressure from conserved variables
   //---------------------------------------------------------------------------
   template <typename number, typename InputVector>
   static
   number
   compute_pressure (const InputVector &W)
   {
      return ((gas_gamma-1.0) *
              (*(W.begin() + energy_component) -
               compute_kinetic_energy<number>(W)));
   }

   //---------------------------------------------------------------------------
   // Compute maximum eigenvalue
   //---------------------------------------------------------------------------
   template <typename InputVector>
   static
   typename InputVector::value_type
   max_eigenvalue (const InputVector &W)
   {
      typedef typename InputVector::value_type number;

      const number pressure = 
         compute_pressure<number> (W);

      number velocity = 0;
      for (unsigned int d=0; d<dim; ++d)
         velocity += *(W.begin()+d) *
                     *(W.begin()+d);
      velocity = std::sqrt(velocity) / (*(W.begin()+density_component));

      return velocity + std::sqrt(gas_gamma * pressure / (*(W.begin()+density_component)));
   }
   
   //---------------------------------------------------------------------------
   // Compute maximum eigenvalue in normal direction
   //---------------------------------------------------------------------------
   template <typename InputVector>
   static
   typename InputVector::value_type
   max_eigenvalue (const InputVector           &W,
                   const dealii::Tensor<1,dim> &normal)
   {
      typedef typename InputVector::value_type number;
      
      const number pressure = compute_pressure<number> (W);
      const number sonic = std::sqrt(gas_gamma * pressure / (*(W.begin()+density_component)));
      
      number velocity = 0;
      for (unsigned int d=0; d<dim; ++d)
         velocity += *(W.begin()+d) * normal[d];
      
      velocity /=  (*(W.begin()+density_component));
      
      return std::fabs(velocity) + sonic;
   }
   
   //---------------------------------------------------------------------------
   // Compute sound speed
   //---------------------------------------------------------------------------
   template <typename InputVector>
   static
   typename InputVector::value_type
   sound_speed (const InputVector &W)
   {
      typedef typename InputVector::value_type number;

      const number pressure = 
         compute_pressure<number> (W);
      return std::sqrt(gas_gamma * pressure / (*(W.begin()+density_component)));
   }
   
   
   //---------------------------------------------------------------------------
   // Compute cartesian components of flux
   //---------------------------------------------------------------------------
   template <typename InputVector, typename number>
   static
   void compute_flux_matrix (const InputVector &W,
                             number            (&flux)[n_components][dim])
   {
      // First compute the pressure that
      // appears in the flux matrix, and
      // then compute the first
      // <code>dim</code> columns of the
      // matrix that correspond to the
      // momentum terms:
      const number pressure = compute_pressure<number> (W);
      
      for (unsigned int d=0; d<dim; ++d)
      {
         for (unsigned int e=0; e<dim; ++e)
            flux[d][e] = W[d] *
                         W[e] /
                         W[density_component];
         
         flux[d][d] += pressure;
      }
      
      // Then the terms for the
      // density (i.e. mass
      // conservation), and,
      // lastly, conservation of
      // energy:
      for (unsigned int d=0; d<dim; ++d)
         flux[density_component][d] = W[d];
      
      for (unsigned int d=0; d<dim; ++d)
         flux[energy_component][d] = W[d] /
                                     W[density_component] *
                                     (W[energy_component] + pressure);
   }
   
   //---------------------------------------------------------------------------
   // Compute flux along normal
   //---------------------------------------------------------------------------
   template <typename InputVector, typename number>
   static
   void normal_flux (const InputVector           &W,
                     const dealii::Tensor<1,dim> &normal,
                     number                      (&flux)[n_components])
   {
      const number pressure = compute_pressure<number> (W);
      
      number vdotn = 0.0;
      for (unsigned int d=0; d<dim; ++d)
         vdotn += W[d] * normal[d];
      vdotn /= W[density_component];
      
      flux[density_component] = W[density_component] * vdotn;
      flux[energy_component] = (W[energy_component] + pressure) * vdotn;
      for (unsigned int d=0; d<dim; ++d)
         flux[d] = pressure * normal[d] + W[d] * vdotn;
   }
   
   //---------------------------------------------------------------------------
   // Left and right eigenvector matrices
   // Lx, Rx = along x direction
   // Ly, Ry = along y direction
   // Expressions taken from
   // http://people.nas.nasa.gov/~pulliam/Classes/New_notes/euler_notes.pdf
   // Note: This is implemented only for 2-D
   //---------------------------------------------------------------------------
   static
   void compute_eigen_matrix (const dealii::Vector<double> &W,
                              double            (&Rx)[n_components][n_components],
                              double            (&Lx)[n_components][n_components],
                              double            (&Ry)[n_components][n_components],
                              double            (&Ly)[n_components][n_components])
   {
      double g1   = gas_gamma - 1.0;
      double rho  = W[density_component];
      double E    = W[energy_component];
      double u    = W[0] / rho;
      double v    = W[1] / rho;
      double q2   = u*u + v*v;
      double p    = g1 * (E - 0.5 * rho * q2);
      double c2   = gas_gamma * p / rho;
      double c    = std::sqrt(c2);
      double beta = 0.5/c2;
      double phi2 = 0.5*g1*q2;
      double h    = c2/g1 + 0.5*q2;
      
      Rx[0][0] = 1;      Rx[0][1] = 0;  Rx[0][2] = 1;     Rx[0][3] = 1;
      Rx[1][0] = u;      Rx[1][1] = 0;  Rx[1][2] = u+c;   Rx[1][3] = u-c;
      Rx[2][0] = v;      Rx[2][1] = -1; Rx[2][2] = v;     Rx[2][3] = v;
      Rx[3][0] = 0.5*q2; Rx[3][1] = -v; Rx[3][2] = h+c*u; Rx[3][3] = h-c*u;
      
      Ry[0][0] = 1;      Ry[0][1] = 0;  Ry[0][2] = 1;     Ry[0][3] = 1;
      Ry[1][0] = u;      Ry[1][1] = 1;  Ry[1][2] = u;     Ry[1][3] = u;
      Ry[2][0] = v;      Ry[2][1] = 0;  Ry[2][2] = v+c;   Ry[2][3] = v-c;
      Ry[3][0] = 0.5*q2; Ry[3][1] = u;  Ry[3][2] = h+c*v; Ry[3][3] = h-c*v;
      
      Lx[0][0] = 1-phi2/c2;       Lx[0][1] = g1*u/c2;       Lx[0][2] = g1*v/c2;    Lx[0][3] = -g1/c2;
      Lx[1][0] = v;               Lx[1][1] = 0;             Lx[1][2] = -1;         Lx[1][3] = 0;
      Lx[2][0] = beta*(phi2-c*u); Lx[2][1] = beta*(c-g1*u); Lx[2][2] = -beta*g1*v; Lx[2][3] = beta*g1;
      Lx[3][0] = beta*(phi2+c*u); Lx[3][1] =-beta*(c+g1*u); Lx[3][2] = -beta*g1*v; Lx[3][3] = beta*g1;
      
      Ly[0][0] = 1-phi2/c2;       Ly[0][1] = g1*u/c2;       Ly[0][2] = g1*v/c2;       Ly[0][3] = -g1/c2;
      Ly[1][0] = -u;              Ly[1][1] = 1;             Ly[1][2] = 0;             Ly[1][3] = 0;
      Ly[2][0] = beta*(phi2-c*v); Ly[2][1] =-beta*g1*u;     Ly[2][2] = beta*(c-g1*v); Ly[2][3] = beta*g1;
      Ly[3][0] = beta*(phi2+c*v); Ly[3][1] =-beta*g1*u;     Ly[3][2] =-beta*(c+g1*v); Ly[3][3] = beta*g1;
      
   }
   
   //---------------------------------------------------------------------------
   // Left and right eigenvector matrices in the direction of (kx,ky)
   // Following function uses the streamline direction.
   // Expressions taken from
   // http://people.nas.nasa.gov/~pulliam/Classes/New_notes/euler_notes.pdf
   // Note: This is implemented only for 2-D
   //---------------------------------------------------------------------------
   static
   void compute_eigen_matrix (const dealii::Vector<double> &W,
                              double            (&R)[n_components][n_components],
                              double            (&L)[n_components][n_components])
   {
      double g1   = gas_gamma - 1.0;
      double rho  = W[density_component];
      double E    = W[energy_component];
      double u    = W[0] / rho;
      double v    = W[1] / rho;
      double q2   = u*u + v*v;
      double p    = g1 * (E - 0.5 * rho * q2);
      double c2   = gas_gamma * p / rho;
      double c    = std::sqrt(c2);
      double beta = 0.5/c2;
      double phi2 = 0.5*g1*q2;
      double h    = c2/g1 + 0.5*q2;
      double theta= atan2(v,u);
      double kx   = cos(theta);
      double ky   = sin(theta);
      double uk   = u*kx + v*ky;
      
      R[0][0] = 1;      R[0][1] = 0;         R[0][2] = 1;      R[0][3] = 1;
      R[1][0] = u;      R[1][1] = ky;        R[1][2] = u+kx*c; R[1][3] = u-kx*c;
      R[2][0] = v;      R[2][1] = -kx;       R[2][2] = v+ky*c; R[2][3] = v-ky*c;
      R[3][0] = 0.5*q2; R[3][1] = ky*u-kx*v; R[3][2] = h+c*uk; R[3][3] = h-c*uk;
      
      L[0][0] = 1-phi2/c2;        L[0][1] = g1*u/c2;          L[0][2] = g1*v/c2;           L[0][3] = -g1/c2;
      L[1][0] =-(ky*u-kx*v);      L[1][1] = ky;               L[1][2] = -kx;               L[1][3] = 0;
      L[2][0] = beta*(phi2-c*uk); L[2][1] = beta*(kx*c-g1*u); L[2][2] =  beta*(ky*c-g1*v); L[2][3] = beta*g1;
      L[3][0] = beta*(phi2+c*uk); L[3][1] =-beta*(kx*c+g1*u); L[3][2] = -beta*(ky*c+g1*v); L[3][3] = beta*g1;
   }
   
   //---------------------------------------------------------------------------
   // convert from conserved to characteristic variables: W = L*W
   //---------------------------------------------------------------------------
   static
   void transform_to_char (const double           (&L)[n_components][n_components],
                           dealii::Vector<double> &W)
   {
      dealii::Vector<double> V(n_components);
      
      V[0] = W[density_component];
      V[n_components-1] = W[energy_component];
      for(unsigned int d=0; d<dim; ++d)
         V[d+1] = W[d];
      
      W = 0;
      for(unsigned int i=0; i<n_components; ++i)
         for(unsigned int j=0; j<n_components; ++j)
            W[i] += L[i][j] * V[j];
   }
   
   //---------------------------------------------------------------------------
   // convert from characteristic to conserved variables: W = R*W
   //---------------------------------------------------------------------------
   static
   void transform_to_con (const double           (&R)[n_components][n_components],
                          dealii::Vector<double> &W)
   {
      dealii::Vector<double> V(n_components);
      
      V = 0;
      for(unsigned int i=0; i<n_components; ++i)
         for(unsigned int j=0; j<n_components; ++j)
            V[i] += R[i][j] * W[j];

      W[density_component] = V[0];
      W[energy_component] = V[n_components-1];
      for(unsigned int d=0; d<dim; ++d)
         W[d] = V[d+1];
      
   }

   // @sect4{EulerEquations::compute_normal_flux}
   
   // On the boundaries of the
   // domain and across hanging
   // nodes we use a numerical flux
   // function to enforce boundary
   // conditions.  This routine is
   // the basic Lax-Friedrich's flux
   // with a stabilization parameter
   // $\alpha$. It's form has also
   // been given already in the
   // introduction:

   // --------------------------------------------------------------------------
   // Local lax-Friedrichs flux
   // --------------------------------------------------------------------------
   template <typename InputVector>
   static
   void lxf_flux 
   (
    const dealii::Tensor<1,dim>      &normal,
    const InputVector                &Wplus,
    const InputVector                &Wminus,
    const dealii::Vector<double>     &Aplus,
    const dealii::Vector<double>     &Aminus,
    typename InputVector::value_type (&normal_flux)[n_components]
   )
   {
      typedef typename InputVector::value_type number;

      // Normal velocity
      number vdotn_plus=0, vdotn_minus=0;
      
      for(unsigned int d=0; d<dim; ++d)
      {
         vdotn_plus  += Wplus[d]  * normal[d];
         vdotn_minus += Wminus[d] * normal[d];
      }
      
      vdotn_plus  /= Wplus [density_component];
      vdotn_minus /= Wminus[density_component];
      
      // pressure
      number p_plus, p_minus;

      p_plus  = compute_pressure<number> (Wplus);
      p_minus = compute_pressure<number> (Wminus);
      
      // Maximum eigenvalue at cell face
      number lambda_plus = max_eigenvalue (Aplus, normal);
      number lambda_minus = max_eigenvalue (Aminus, normal);
      number lambda = std::max(lambda_plus, lambda_minus);
      
      // Momentum flux
      for (unsigned int d=0; d<dim; ++d)
         normal_flux[d] = 0.5 * ( p_plus  * normal[d] + Wplus [d] * vdotn_plus +
                                  p_minus * normal[d] + Wminus[d] * vdotn_minus );

      // Density flux
      normal_flux[density_component] = 0.5 * (Wplus [density_component] * vdotn_plus +
                                              Wminus[density_component] * vdotn_minus);
      
      // Energy flux
      normal_flux[energy_component] = 0.5 * ((Wplus [energy_component] + p_plus)  * vdotn_plus +
                                             (Wminus[energy_component] + p_minus) * vdotn_minus);
      
      // Dissipation flux
      for (unsigned int c=0; c<n_components; ++c)
         normal_flux[c] += 0.5 * lambda * (Wplus[c] - Wminus[c]);
   }
   
   // --------------------------------------------------------------------------
   // Steger-Warming flux
   // --------------------------------------------------------------------------
   template <typename InputVector>
   static
   void steger_warming_flux 
   (
    const dealii::Tensor<1,dim>      &normal,
    const InputVector                &Wplus,
    const InputVector                &Wminus,
    typename InputVector::value_type (&normal_flux)[n_components]
   )
   {
      typedef typename InputVector::value_type number;

      number pflux[n_components], mflux[n_components];
      
      // normal velocity and velocity magnitude
      number vdotn_plus=0, vdotn_minus=0, q2_plus=0, q2_minus=0;

      for(unsigned int d=0; d<dim; ++d)
      {
         vdotn_plus  += Wplus[d]  * normal[d];
         vdotn_minus += Wminus[d] * normal[d];
         
         q2_plus  += Wplus[d]  * Wplus[d];
         q2_minus += Wminus[d] * Wminus[d];
      }
      
      vdotn_plus  /= Wplus [density_component];
      vdotn_minus /= Wminus[density_component];
      
      q2_plus  /= Wplus [density_component] * Wplus [density_component];
      q2_minus /= Wminus[density_component] * Wminus[density_component];
      
      // pressure
      number p_plus, p_minus;
      
      p_plus  = compute_pressure<number> (Wplus);
      p_minus = compute_pressure<number> (Wminus);
      
      // sound speed
      number c_plus, c_minus;
      c_plus  = std::sqrt(gas_gamma * p_plus  / Wplus [density_component]);
      c_minus = std::sqrt(gas_gamma * p_minus / Wminus[density_component]);

      // positive flux
      number l1p, l2p, l3p, ap, fp;
      
      l1p = std::max( vdotn_plus,          0.0);
      l2p = std::max( vdotn_plus + c_plus, 0.0);
      l3p = std::max( vdotn_plus - c_plus, 0.0);
      ap  = 2.0 * (gas_gamma - 1.0) * l1p + l2p + l3p;
      fp  = 0.5 * Wplus[density_component] / gas_gamma;
      
      for(unsigned int d=0; d<dim; ++d)
         pflux[d] = ap * Wplus[d]/Wplus[density_component] +
                          c_plus * (l2p - l3p) * normal[d];
      
      pflux[density_component] = ap;
      pflux[energy_component] = 0.5 * ap * q2_plus +
                                c_plus * vdotn_plus * (l2p - l3p) +
                                c_plus * c_plus * (l2p + l3p) / (gas_gamma - 1.0);
      
      // negative flux
      number l1m, l2m, l3m, am, fm;
      
      l1m = std::min( vdotn_minus,           0.0);
      l2m = std::min( vdotn_minus + c_minus, 0.0);
      l3m = std::min( vdotn_minus - c_minus, 0.0);
      am  = 2.0 * (gas_gamma - 1.0) * l1m + l2m + l3m;
      fm  = 0.5 * Wminus[density_component] / gas_gamma;
      
      for(unsigned int d=0; d<dim; ++d)
         mflux[d] = am * Wminus[d]/Wminus[density_component] +
                    c_minus * (l2m - l3m) * normal[d];
      
      mflux[density_component] = am;
      mflux[energy_component] = 0.5 * am * q2_minus +
                                c_minus * vdotn_minus * (l2m - l3m) +
                                c_minus * c_minus * (l2m + l3m) / (gas_gamma - 1.0);
            
      // Total flux
      for (unsigned int c=0; c<n_components; ++c)
         normal_flux[c] = fp * pflux[c] + fm * mflux[c];
   }
   
   // --------------------------------------------------------------------------
   // Roe flux
   // --------------------------------------------------------------------------
   template <typename InputVector>
   static
   void roe_flux
   (
    const dealii::Tensor<1,dim>      &normal,
    const InputVector                &W_l,
    const InputVector                &W_r,
    typename InputVector::value_type (&normal_flux)[n_components]
    )
   {
      typedef typename InputVector::value_type number;
      
      number rho_l_sqrt = std::sqrt(W_l[density_component]);
      number rho_r_sqrt = std::sqrt(W_r[density_component]);
      number fact_l = rho_l_sqrt / (rho_l_sqrt + rho_r_sqrt);
      number fact_r = 1.0 - fact_l;
      
      number v_l[dim], v_r[dim], velocity[dim], dv[dim];
      number v2_l = 0, v2_r = 0;
      number v_l_normal = 0, v_r_normal = 0;
      number vel_normal = 0, v2 = 0;
      number v_dot_dv = 0;
      for(unsigned int d=0; d<dim; ++d)
      {
         v_l[d]      = W_l[d] / W_l[density_component];
         v_r[d]      = W_r[d] / W_r[density_component];
         v2_l       += v_l[d] * v_l[d];
         v2_r       += v_r[d] * v_r[d];
         v_l_normal += v_l[d] * normal[d];
         v_r_normal += v_r[d] * normal[d];
         
         velocity[d] = v_l[d] * fact_l + v_r[d] * fact_r;
         vel_normal += velocity[d] * normal[d];
         v2         += velocity[d] * velocity[d];
         dv[d]       = v_r[d] - v_l[d];
         v_dot_dv   += velocity[d] * dv[d];
      }
      
      number p_l = (gas_gamma-1) * (W_l[energy_component] - 0.5 * W_l[density_component] * v2_l);
      number p_r = (gas_gamma-1) * (W_r[energy_component] - 0.5 * W_r[density_component] * v2_r);
      
      number h_l = gas_gamma * p_l / W_l[density_component] / (gas_gamma-1) + 0.5 * v2_l;
      number h_r = gas_gamma * p_r / W_r[density_component] / (gas_gamma-1) + 0.5 * v2_r;
      
      number density = rho_l_sqrt * rho_r_sqrt;
      number h = h_l * fact_l + h_r * fact_r;
      number c = std::sqrt( (gas_gamma-1.0) * (h - 0.5*v2) );
      number drho = W_r[density_component] - W_l[density_component];
      number dp = p_r - p_l;
      number dvn = v_r_normal - v_l_normal;
      
      number a1 = (dp - density * c * dvn) / (2.0*c*c);
      number a2 = drho - dp / (c*c);
      number a3 = (dp + density * c * dvn) / (2.0*c*c);

      number l1 = std::fabs(vel_normal - c);
      number l2 = std::fabs(vel_normal);
      number l3 = std::fabs(vel_normal + c);

      // entropy fix
      number delta = 0.1 * c;
      if(l1 < delta) l1 = 0.5 * (l1*l1/delta + delta);
      if(l3 < delta) l3 = 0.5 * (l3*l3/delta + delta);
      
      number Dflux[n_components];
      Dflux[density_component] = l1 * a1 + l2 * a2 + l3 * a3;
      Dflux[energy_component] = l1 * a1 * (h - c * vel_normal)
                              + l2 * a2 * 0.5 * v2
                              + l2 * density * (v_dot_dv - vel_normal * dvn)
                              + l3 * a3 * (h + c * vel_normal);
      normal_flux[density_component] = 0.5 * (W_l[density_component] * v_l_normal +
                                              W_r[density_component] * v_r_normal
                                              - Dflux[density_component]);
      normal_flux[energy_component] = 0.5 * (W_l[density_component] * h_l * v_l_normal +
                                             W_r[density_component] * h_r * v_r_normal
                                              - Dflux[energy_component]);
      number p_avg = 0.5 * (p_l + p_r);
      for(unsigned int d=0; d<dim; ++d)
      {
         Dflux[d] = (velocity[d] - normal[d] * c) * l1 * a1
                  + velocity[d] * l2 * a2
                  + (dv[d] - normal[d] * dvn) * l2 * density
                  + (velocity[d] + normal[d] * c) * l3 * a3;
         normal_flux[d] = normal[d] * p_avg
                        + 0.5 * (W_l[d] * v_l_normal + W_r[d] * v_r_normal)
                        - 0.5 * Dflux[d];
      }
   }
   
   
   // --------------------------------------------------------------------------
   // HLLC flux
   // Code borrowed from SU2 v2.0.2
   // --------------------------------------------------------------------------
   template <typename InputVector>
   static
   void hllc_flux
   (
    const dealii::Tensor<1,dim>      &normal,
    const InputVector                &W_l,
    const InputVector                &W_r,
    typename InputVector::value_type (&normal_flux)[n_components]
    )
   {
      typedef typename InputVector::value_type number;
      
      number rho_l_sqrt = std::sqrt(W_l[density_component]);
      number rho_r_sqrt = std::sqrt(W_r[density_component]);
      number fact_l = rho_l_sqrt / (rho_l_sqrt + rho_r_sqrt);
      number fact_r = 1.0 - fact_l;
      
      number v_l[dim], v_r[dim], velocity[dim];
      number v2_l = 0, v2_r = 0;
      number v_l_normal = 0, v_r_normal = 0;
      number vel_normal = 0, v2 = 0;
      for(unsigned int d=0; d<dim; ++d)
      {
         v_l[d]      = W_l[d] / W_l[density_component];
         v_r[d]      = W_r[d] / W_r[density_component];
         v2_l       += v_l[d] * v_l[d];
         v2_r       += v_r[d] * v_r[d];
         v_l_normal += v_l[d] * normal[d];
         v_r_normal += v_r[d] * normal[d];
         
         velocity[d] = v_l[d] * fact_l + v_r[d] * fact_r;
         vel_normal += velocity[d] * normal[d];
         v2         += velocity[d] * velocity[d];
      }
      
      //pressure
      number p_l = (gas_gamma-1) * (W_l[energy_component] - 0.5 * W_l[density_component] * v2_l);
      number p_r = (gas_gamma-1) * (W_r[energy_component] - 0.5 * W_r[density_component] * v2_r);
      
      // enthalpy
      number h_l = (W_l[energy_component] + p_l) / W_l[density_component];
      number h_r = (W_r[energy_component] + p_r) / W_r[density_component];

      // sound speed
      number c_l = std::sqrt(gas_gamma * p_l / W_l[density_component]);
      number c_r = std::sqrt(gas_gamma * p_r / W_r[density_component]);
      
      // energy per unit mass
      number e_l = W_l[energy_component] / W_l[density_component];
      number e_r = W_r[energy_component] / W_r[density_component];
      
      // roe average
      number h = h_l * fact_l + h_r * fact_r;
      number c = std::sqrt( (gas_gamma-1.0) * (h - 0.5*v2) );
      
      // speed of sound at l and r
      number s_l = std::min(vel_normal-c, v_l_normal-c_l);
      number s_r = std::max(vel_normal+c, v_r_normal+c_r);

      // speed of contact
      number s_m = (p_l - p_r
                    - W_l[density_component] * v_l_normal * (s_l-v_l_normal)
                    + W_r[density_component] * v_r_normal * (s_r-v_r_normal))
      /(W_r[density_component]*(s_r-v_r_normal) - W_l[density_component]*(s_l-v_l_normal));
      
      // Pressure at right and left (Pressure_j=Pressure_i) side of contact surface
      number pStar = W_r[density_component] * (v_r_normal-s_r)*(v_r_normal-s_m) + p_r;

      if (s_m >= 0.0) {
         if (s_l > 0.0)
         {
            normal_flux[density_component] = W_l[density_component]*v_l_normal;
            for (unsigned int d = 0; d < dim; d++)
               normal_flux[d] = W_l[density_component]*v_l[d]*v_l_normal + p_l*normal[d];
            normal_flux[energy_component] = e_l*W_l[density_component]*v_l_normal + p_l*v_l_normal;
         }
         else
         {
            number invSLmSs = 1.0/(s_l-s_m);
            number sLmuL = s_l-v_l_normal;
            number rhoSL = W_l[density_component]*sLmuL*invSLmSs;
            number rhouSL[dim];
            for (unsigned int d = 0; d < dim; d++)
               rhouSL[d] = (W_l[density_component]*v_l[d]*sLmuL+(pStar-p_l)*normal[d])*invSLmSs;
            number eSL = (sLmuL*e_l*W_l[density_component]-p_l*v_l_normal+pStar*s_m)*invSLmSs;
            
            normal_flux[density_component] = rhoSL*s_m;
            for (unsigned int d = 0; d < dim; d++)
               normal_flux[d] = rhouSL[d]*s_m + pStar*normal[d];
            normal_flux[energy_component] = (eSL+pStar)*s_m;
         }
      }
      else
      {
         if (s_r >= 0.0)
         {
            number invSRmSs = 1.0/(s_r-s_m);
            number sRmuR = s_r-v_r_normal;
            number rhoSR = W_r[density_component]*sRmuR*invSRmSs;
            number rhouSR[dim];
            for (unsigned int d = 0; d < dim; d++)
               rhouSR[d] = (W_r[density_component]*v_r[d]*sRmuR+(pStar-p_r)*normal[d])*invSRmSs;
            number eSR = (sRmuR*e_r*W_r[density_component]-p_r*v_r_normal+pStar*s_m)*invSRmSs;
            
            normal_flux[density_component] = rhoSR*s_m;
            for (unsigned int d = 0; d < dim; d++)
               normal_flux[d] = rhouSR[d]*s_m + pStar*normal[d];
            normal_flux[energy_component] = (eSR+pStar)*s_m;
         }
         else
         {
            normal_flux[density_component] = W_r[density_component]*v_r_normal;
            for (unsigned int d = 0; d < dim; d++)
               normal_flux[d] = W_r[density_component]*v_r[d]*v_r_normal + p_r*normal[d];
            normal_flux[energy_component] = e_r*W_r[density_component]*v_r_normal + p_r*v_r_normal;
         }
      }
      
   }
   // --------------------------------------------------------------------------
   // Compute dissipation matrix in entropy stable flux
   // --------------------------------------------------------------------------
   static
   void kep_diff_matrix(const dealii::Tensor<1,dim>  &normal,
                        const dealii::Vector<double> &W_l,
                        const dealii::Vector<double> &W_r,
                        double (&Dm)[n_components][n_components])
   {
      static const double BETA = 1.0/6.0;

      double rhol = W_l[density_component];
      double rhor = W_r[density_component];
      double rho = logavg( rhol, rhor );
      
      double v_l[dim], v_r[dim], vel[dim];
      double v2_l = 0, v2_r = 0;
      double vnl = 0, vnr = 0;
      double vel_normal = 0, v2 = 0;
      for(unsigned int d=0; d<dim; ++d)
      {
         v_l[d]  = W_l[d] / W_l[density_component];
         v_r[d]  = W_r[d] / W_r[density_component];
         v2_l   += v_l[d] * v_l[d];
         v2_r   += v_r[d] * v_r[d];
         vnl    += v_l[d] * normal[d];
         vnr    += v_r[d] * normal[d];
         
         vel[d]      = 0.5 * (v_l[d] + v_r[d]);
         vel_normal += vel[d] * normal[d];
         v2         += vel[d] * vel[d];
      }
      
      double vel2 = 0.5 * (v2_l + v2_r);
      
      //pressure
      double p_l = (gas_gamma-1) * (W_l[energy_component] - 0.5 * W_l[density_component] * v2_l);
      double p_r = (gas_gamma-1) * (W_r[energy_component] - 0.5 * W_r[density_component] * v2_r);
      
      double betal = 0.5 * rhol / p_l;
      double betar = 0.5 * rhor / p_r;
      double beta  = logavg(betal, betar);
      
      double a     = sqrt(0.5 * gas_gamma / beta);
      double p     = 0.5 * (rhol + rhor) / (betal + betar);
      
      // entropy dissipation
      // eigenvectors
      double H  = a*a/(gas_gamma-1.0) + 0.5*v2;
      double v1 = vel[0] * normal[1] - vel[1] * normal[0];
      double R[][4] = {
         {            1,             1,     0,              1                },
         {vel[0] - a*normal[0],   vel[0],   normal[1],  vel[0] + a*normal[0] },
         {vel[1] - a*normal[1],   vel[1],  -normal[0],  vel[1] + a*normal[1] },
         {H      - a*vel_normal,  0.5*v2,   v1,         H      + a*vel_normal}
      };
      
      // eigenvalues
      double al  = sqrt (gas_gamma * p_l / rhol);
      double ar  = sqrt (gas_gamma * p_r / rhor);
      double LambdaL[] = { vnl - al, vnl, vnl, vnl + al };
      double LambdaR[] = { vnr - ar, vnr, vnr, vnr + ar };
      double l2, l3;
      l2 = l3 = fabs(vel_normal);
      double Lambda[]  = { fabs(vel_normal - a) + BETA*fabs(LambdaL[0]-LambdaR[0]),
                           l2,
                           l3,
                           fabs(vel_normal + a) + BETA*fabs(LambdaL[3]-LambdaR[3])};
      
      double S[] = { 0.5*rho/gas_gamma, (gas_gamma-1.0)*rho/gas_gamma, p, 0.5*rho/gas_gamma };
      double D[] = { Lambda[0]*S[0],
                     Lambda[1]*S[1],
                     Lambda[2]*S[2],
                     Lambda[3]*S[3]
                   };
      
      // Symmetric diffusion matrix
      for(int i=0; i<4; ++i)
      {
         for(int j=0; j<i; ++j)
            Dm[i][j] = Dm[j][i];
         
         for(int j=i; j<4; ++j)
         {
            Dm[i][j] = 0;
            for(int k=0; k<4; ++k)
               Dm[i][j] += R[i][k] * D[k] * R[j][k];
         }
      }
   }
   
   // --------------------------------------------------------------------------
   // My kinetic energy preserving and entropy stable flux
   // Written only for 2-D case
   // --------------------------------------------------------------------------
   template <typename InputVector>
   static
   void kep_flux
   (
    const dealii::Tensor<1,dim>      &normal,
    const InputVector                &W_l,
    const InputVector                &W_r,
    const dealii::Vector<double>     &Aplus,
    const dealii::Vector<double>     &Aminus,
    typename InputVector::value_type (&normal_flux)[n_components]
    )
   {
      typedef typename InputVector::value_type number;
      
      number rhol = W_l[density_component];
      number rhor = W_r[density_component];
      number rho = logavg( rhol, rhor );
      
      number v_l[dim], v_r[dim], vel[dim];
      number v2_l = 0, v2_r = 0;
      number vnl = 0, vnr = 0;
      number vel_normal = 0, v2 = 0;
      for(unsigned int d=0; d<dim; ++d)
      {
         v_l[d]  = W_l[d] / W_l[density_component];
         v_r[d]  = W_r[d] / W_r[density_component];
         v2_l   += v_l[d] * v_l[d];
         v2_r   += v_r[d] * v_r[d];
         vnl    += v_l[d] * normal[d];
         vnr    += v_r[d] * normal[d];
         
         vel[d]      = 0.5 * (v_l[d] + v_r[d]);
         vel_normal += vel[d] * normal[d];
         v2         += vel[d] * vel[d];
      }
      
      number vel2 = 0.5 * (v2_l + v2_r);
      
      //pressure
      number p_l = (gas_gamma-1) * (W_l[energy_component] - 0.5 * W_l[density_component] * v2_l);
      number p_r = (gas_gamma-1) * (W_r[energy_component] - 0.5 * W_r[density_component] * v2_r);
      
      number betal = 0.5 * rhol / p_l;
      number betar = 0.5 * rhor / p_r;
      number beta  = logavg(betal, betar);
      
      number a     = sqrt(0.5 * gas_gamma / beta);
      number p     = 0.5 * (rhol + rhor) / (betal + betar);
      
      // central flux
      normal_flux[density_component] = rho * vel_normal;
      for(unsigned int d=0; d<dim;++d)
         normal_flux[d] = normal[d] * p + vel[d] * normal_flux[density_component];
      normal_flux[energy_component] =
         0.5 * ( 1.0/((gas_gamma-1.0)*beta) - vel2) * normal_flux[density_component]
         + normal_flux[0] * vel[0] + normal_flux[1] * vel[1];
      
      number Dm[n_components][n_components];
      kep_diff_matrix(normal, Aplus, Aminus, Dm);
      
      // jump in entropy: s = log(p) - gamma*log(rho)
      number ds    = log(p_r/p_l) - gas_gamma * log(rhor/rhol);
      // Jump in entropy variables
      number dV[] = { -ds/(gas_gamma-1.0) - (betar*v2_r - betal*v2_l),
                       2.0*(betar*v_r[0] - betal*v_l[0]),
                       2.0*(betar*v_r[1] - betal*v_l[1]),
                      -2.0*(betar - betal) };
      
      // diffusive flux = R * Lambda * S * R^T * dV
      number Diff[] = {0.0, 0.0, 0.0, 0.0};
      for(unsigned int i=0; i<4; ++i)
         for(unsigned int j=0; j<4; ++j)
               Diff[i] += Dm[i][j] * dV[j];
      
      normal_flux[density_component] -= 0.5 * Diff[0];
      normal_flux[0]                 -= 0.5 * Diff[1];
      normal_flux[1]                 -= 0.5 * Diff[2];
      normal_flux[energy_component]  -= 0.5 * Diff[3];
   }
   
   // --------------------------------------------------------------------------
   // My kinetic energy preserving and entropy stable flux
   // Written only for 2-D case
   // --------------------------------------------------------------------------
   template <typename InputVector>
   static
   void kep_flux2
   (
    const dealii::Tensor<1,dim>      &normal,
    const InputVector                &W_l,
    const InputVector                &W_r,
    const dealii::Vector<double>     &Aplus,
    const dealii::Vector<double>     &Aminus,
    typename InputVector::value_type (&normal_flux)[n_components]
    )
   {
      typedef typename InputVector::value_type number;

      static const double BETA = 1.0/6.0;
      
      number rhol = W_l[density_component];
      number rhor = W_r[density_component];
      number rho = logavg( rhol, rhor );
      
      number v_l[dim], v_r[dim], vel[dim];
      number v2_l = 0, v2_r = 0;
      number vnl = 0, vnr = 0;
      number vel_normal = 0, v2 = 0;
      for(unsigned int d=0; d<dim; ++d)
      {
         v_l[d]  = W_l[d] / W_l[density_component];
         v_r[d]  = W_r[d] / W_r[density_component];
         v2_l   += v_l[d] * v_l[d];
         v2_r   += v_r[d] * v_r[d];
         vnl    += v_l[d] * normal[d];
         vnr    += v_r[d] * normal[d];
         
         vel[d]      = 0.5 * (v_l[d] + v_r[d]);
         vel_normal += vel[d] * normal[d];
         v2         += vel[d] * vel[d];
      }
      
      number vel2 = 0.5 * (v2_l + v2_r);
      
      //pressure
      number p_l = (gas_gamma-1) * (W_l[energy_component] - 0.5 * W_l[density_component] * v2_l);
      number p_r = (gas_gamma-1) * (W_r[energy_component] - 0.5 * W_r[density_component] * v2_r);
      
      number betal = 0.5 * rhol / p_l;
      number betar = 0.5 * rhor / p_r;
      number beta  = logavg(betal, betar);
      
      number a     = sqrt(0.5 * gas_gamma / beta);
      number p     = 0.5 * (rhol + rhor) / (betal + betar);
      
      // central flux
      normal_flux[density_component] = rho * vel_normal;
      for(unsigned int d=0; d<dim;++d)
         normal_flux[d] = normal[d] * p + vel[d] * normal_flux[density_component];
      normal_flux[energy_component] =
           0.5 * ( 1.0/((gas_gamma-1.0)*beta) - vel2) * normal_flux[density_component]
         + normal_flux[0] * vel[0] + normal_flux[1] * vel[1];
      
      // entropy dissipation
      // eigenvectors
      number H  = a*a/(gas_gamma-1.0) + 0.5*v2;
      number v1 = vel[0] * normal[1] - vel[1] * normal[0];
      number R[][4] = {
         {            1,             1,     0,              1                },
         {vel[0] - a*normal[0],   vel[0],   normal[1],  vel[0] + a*normal[0] },
         {vel[1] - a*normal[1],   vel[1],  -normal[0],  vel[1] + a*normal[1] },
         {H      - a*vel_normal,  0.5*v2,   v1,         H      + a*vel_normal}
      };
      
      // eigenvalues
      number al  = sqrt (gas_gamma * p_l / rhol);
      number ar  = sqrt (gas_gamma * p_r / rhor);
      number LambdaL[] = { vnl - al, vnl, vnl, vnl + al };
      number LambdaR[] = { vnr - ar, vnr, vnr, vnr + ar };
      number l2, l3;
      l2 = l3 = fabs(vel_normal);
      number Lambda[]  = { fabs(vel_normal - a) + BETA*fabs(LambdaL[0]-LambdaR[0]),
                           l2,
                           l3,
                           fabs(vel_normal + a) + BETA*fabs(LambdaL[3]-LambdaR[3])};
      
      number S[] = { 0.5*rho/gas_gamma, (gas_gamma-1.0)*rho/gas_gamma, p, 0.5*rho/gas_gamma };
      number D[] = { Lambda[0]*S[0],
                     Lambda[1]*S[1],
                     Lambda[2]*S[2],
                     Lambda[3]*S[3]
                   };
      
      // jump in entropy: s = log(p) - gamma*log(rho)
      number ds    = log(p_r/p_l) - gas_gamma * log(rhor/rhol);
      // Jump in entropy variables
      number dV[] = { -ds/(gas_gamma-1.0) - (betar*v2_r - betal*v2_l),
                       2.0*(betar*v_r[0] - betal*v_l[0]),
                       2.0*(betar*v_r[1] - betal*v_l[1]),
                      -2.0*(betar - betal) };
      
      // DRT = D * R^T
      number DRT[4][4];
      for(unsigned int i=0; i<4; ++i)
         for(unsigned int j=0; j<4; ++j)
            DRT[i][j] = D[i]*R[j][i];
      
      // diffusive flux = R * Lambda * S * R^T * dV
      number Diff[] = {0.0, 0.0, 0.0, 0.0};
      for(unsigned int i=0; i<4; ++i)
         for(unsigned int j=0; j<4; ++j)
            for(unsigned int k=0; k<4; ++k)
               Diff[i] += R[i][j] * DRT[j][k] * dV[k];
      
      normal_flux[density_component] -= 0.5 * Diff[0];
      normal_flux[0]                 -= 0.5 * Diff[1];
      normal_flux[1]                 -= 0.5 * Diff[2];
      normal_flux[energy_component]  -= 0.5 * Diff[3];
   }

   // --------------------------------------------------------------------------
   // Error function
   // --------------------------------------------------------------------------
   template <typename number>
   static
   number ERF(number xarg)
   {
      // constants
      const double a1 =  0.254829592;
      const double a2 = -0.284496736;
      const double a3 =  1.421413741;
      const double a4 = -1.453152027;
      const double a5 =  1.061405429;
      const double p  =  0.3275911;

      // Save the sign of x
      int sign = 1;
      if (xarg < 0)
         sign = -1;
      number x = std::fabs(xarg);

      // A&S formula 7.1.26
      number t = 1.0/(1.0 + p*x);
      number y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

      return sign * y;
   }
   
   // --------------------------------------------------------------------------
   // Kinetic split fluxes
   // --------------------------------------------------------------------------
   template <typename InputVector>
   static
   void kinetic_split_flux
   (
    int                               sign,
    const dealii::Tensor<1,dim>      &normal,
    const InputVector                &W,
    typename InputVector::value_type (&normal_flux)[n_components]
   )
   {
      typedef typename InputVector::value_type number;
      
      // normal velocity
      number vdotn=0;

      for(unsigned int d=0; d<dim; ++d)
         vdotn += W[d] * normal[d];
      
      vdotn  /= W[density_component];
      
      // pressure
      number pressure, beta, s, A, B, ufact;
      
      pressure  = compute_pressure<number> (W);
      beta      = 0.5 * W[density_component] / pressure;
      s         = vdotn * std::sqrt(beta);
      A         = 0.5 * (1.0 + sign * ERF(s));
      B         = 0.5 * sign * std::exp(-s*s) / std::sqrt(M_PI * beta);
      ufact     = vdotn * A + B;
      
      for(unsigned int d=0; d<dim; ++d)
         normal_flux[d] = pressure * normal[d] * A + W[d] * ufact;
      
      normal_flux[density_component] = W[density_component] * ufact;
      normal_flux[energy_component]  = (W[energy_component] + pressure) * vdotn * A +
                                       (W[energy_component] + 0.5 * pressure) * B;
      
   }

   // --------------------------------------------------------------------------
   // KFVS flux of Deshpande and Mandal
   // --------------------------------------------------------------------------
   template <typename InputVector>
   static
   void kfvs_flux 
   (
    const dealii::Tensor<1,dim>       &normal,
    const InputVector                 &Wplus,
    const InputVector                 &Wminus,
    typename InputVector::value_type  (&normal_flux)[n_components]
   )
   {
      typedef typename InputVector::value_type number;

      number pflux[n_components], mflux[n_components];

      kinetic_split_flux (+1,
                          normal,
                          Wplus,
                          pflux);

      kinetic_split_flux (-1,
                          normal,
                          Wminus,
                          mflux);

      for (unsigned int c=0; c<n_components; ++c)
         normal_flux[c] = pflux[c] + mflux[c];
   }
   
   // --------------------------------------------------------------------------
   // Flux on slip walls. Only pressure flux is present
   // --------------------------------------------------------------------------
   template <typename InputVector>
   static
   void no_penetration_flux 
   (
    const dealii::Tensor<1,dim>      &normal,
    const InputVector                &Wminus,
    typename InputVector::value_type (&normal_flux)[n_components]
   )
   {
      typedef typename InputVector::value_type number;

      // pressure
      number pressure = compute_pressure<number> (Wminus);
      
      for (unsigned int c=0; c<n_components; ++c)
         normal_flux[c] = 0.0;
      
      // Only pressure flux is present
      for (unsigned int c=0; c<dim; ++c)
         normal_flux[c] = pressure * normal[c];
   }
   
   //---------------------------------------------------------------------------
   // @sect4{EulerEquations::compute_forcing_vector}
   //---------------------------------------------------------------------------
   
   // In the same way as describing the flux
   // function $\mathbf F(\mathbf w)$, we
   // also need to have a way to describe
   // the right hand side forcing term. As
   // mentioned in the introduction, we
   // consider only gravity here, which
   // leads to the specific form $\mathbf
   // G(\mathbf w) = \left(
   // g_1\rho, g_2\rho, g_3\rho, 0,
   // \rho \mathbf g \cdot \mathbf v
   // \right)^T$, shown here for
   // the 3d case. More specifically, we
   // will consider only $\mathbf
   // g=(0,0,-1)^T$ in 3d, or $\mathbf
   // g=(0,-1)^T$ in 2d. This naturally
   // leads to the following function:
   template <typename InputVector, typename number>
   static
   void compute_forcing_vector (const InputVector &W,
                                const dealii::Vector<double> &ext_force,
                                number            (&forcing)[n_components])
   {
      forcing[density_component] = 0.0;
      forcing[energy_component] = 0.0;
      
      for(int d=0; d<dim; ++d)
      {
         forcing[d] = W[density_component] * ext_force[d];
         forcing[energy_component] += W[d] * ext_force[d];
      }
   }
   
   
   //---------------------------------------------------------------------------
   // @sect4{Dealing with boundary conditions}
   //---------------------------------------------------------------------------
   
   // Another thing we have to deal with is
   // boundary conditions. To this end, let
   // us first define the kinds of boundary
   // conditions we currently know how to
   // deal with:
   enum BoundaryKind
   {
      inflow_boundary,
      outflow_boundary,
      no_penetration_boundary,
      pressure_boundary,
      farfield_boundary,
      periodic
   };
   
   
   // The next part is to actually decide
   // what to do at each kind of
   // boundary. To this end, remember from
   // the introduction that boundary
   // conditions are specified by choosing a
   // value $\mathbf w^-$ on the outside of
   // a boundary given an inhomogeneity
   // $\mathbf j$ and possibly the
   // solution's value $\mathbf w^+$ on the
   // inside. Both are then passed to the
   // numerical flux $\mathbf
   // H(\mathbf{w}^+, \mathbf{w}^-,
   // \mathbf{n})$ to define boundary
   // contributions to the bilinear form.
   //
   // Boundary conditions can in some cases
   // be specified for each component of the
   // solution vector independently. For
   // example, if component $c$ is marked
   // for inflow, then $w^-_c = j_c$. If it
   // is an outflow, then $w^-_c =
   // w^+_c$. These two simple cases are
   // handled first in the function below.
   //
   // There is a little snag that makes this
   // function unpleasant from a C++
   // language viewpoint: The output vector
   // <code>Wminus</code> will of course be
   // modified, so it shouldn't be a
   // <code>const</code> argument. Yet it is
   // in the implementation below, and needs
   // to be in order to allow the code to
   // compile. The reason is that we call
   // this function at a place where
   // <code>Wminus</code> is of type
   // <code>Table@<2,Sacado::Fad::DFad@<double@>
   // @></code>, this being 2d table with
   // indices representing the quadrature
   // point and the vector component,
   // respectively. We call this function
   // with <code>Wminus[q]</code> as last
   // argument; subscripting a 2d table
   // yields a temporary accessor object
   // representing a 1d vector, just what we
   // want here. The problem is that a
   // temporary accessor object can't be
   // bound to a non-const reference
   // argument of a function, as we would
   // like here, according to the C++ 1998
   // and 2003 standards (something that
   // will be fixed with the next standard
   // in the form of rvalue references).  We
   // get away with making the output
   // argument here a constant because it is
   // the <i>accessor</i> object that's
   // constant, not the table it points to:
   // that one can still be written to. The
   // hack is unpleasant nevertheless
   // because it restricts the kind of data
   // types that may be used as template
   // argument to this function: a regular
   // vector isn't going to do because that
   // one can not be written to when marked
   // <code>const</code>. With no good
   // solution around at the moment, we'll
   // go with the pragmatic, even if not
   // pretty, solution shown here:
   template <typename DataVector>
   static
   void
   compute_Wminus (const BoundaryKind           &boundary_kind,
                   const dealii::Tensor<1,dim>  &normal_vector,
                   const DataVector             &Wplus,
                   const dealii::Vector<double> &boundary_values,
                   const DataVector             &Wminus)
   {
      switch (boundary_kind)
      {
	      case inflow_boundary:
	      {
            for (unsigned int c = 0; c < n_components; ++c)
                  Wminus[c] = boundary_values(c);
            break;
	      }
            
	      case outflow_boundary:
	      {
            for (unsigned int c = 0; c < n_components; ++c)
               Wminus[c] = Wplus[c];
            break;
	      }
            
            // Prescribed pressure boundary
            // conditions are a bit more
            // complicated by the fact that
            // even though the pressure is
            // prescribed, we really are
            // setting the energy component
            // here, which will depend on
            // velocity and pressure. So
            // even though this seems like
            // a Dirichlet type boundary
            // condition, we get
            // sensitivities of energy to
            // velocity and density (unless
            // these are also prescribed):
	      case pressure_boundary:
	      {
            const typename DataVector::value_type
            density = Wplus[density_component];
            
            typename DataVector::value_type kinetic_energy = 0;
            for (unsigned int d=0; d<dim; ++d)
                  kinetic_energy += Wplus[d]*Wplus[d];
            kinetic_energy *= 0.5/density;
            
            for (unsigned int c = 0; c < dim; ++c)
               Wminus[c] = Wplus[c];

            Wminus[density_component] = density;
            Wminus[energy_component] = boundary_values(energy_component) / (gas_gamma-1.0) +
                        kinetic_energy;
            
            break;
	      }
            
	      case no_penetration_boundary:
	      {
            // We prescribe the
            // velocity (we are dealing with a
            // particular component here so
            // that the average of the
            // velocities is orthogonal to the
            // surface normal.  This creates
            // sensitivies of across the
            // velocity components.
            typename DataVector::value_type
               vdotn = 0;
            for (unsigned int d = 0; d < dim; d++)
               vdotn += Wplus[d]*normal_vector[d];
            
            for (unsigned int c = 0; c < dim; ++c)
            {
               Wminus[c] = Wplus[c] - 2.0 * vdotn * normal_vector[c];
            }

            Wminus[density_component] = Wplus[density_component];
            Wminus[energy_component]  = Wplus[energy_component];
            break;
	      }
            
	      case farfield_boundary:
	      {
            for (unsigned int c = 0; c < n_components; ++c)
                  Wminus[c] = boundary_values(c);
            break;
	      }
            
	      default:
            Assert (false, dealii::ExcNotImplemented());
      }
   }
   
   
   //---------------------------------------------------------------------------
   // Compute entropy variables V, given conserved variables W
   //---------------------------------------------------------------------------
   template <typename InputVector, typename number>
   static
   void entropy_var (const InputVector &W,
                     number            (&V)[n_components])
   {
      number pressure = compute_pressure<number> (W);
      number T = pressure / W[density_component];

      number u2 = 0;
      for(unsigned int d=0; d<dim; ++d)
      {
         number u = W[d] / W[density_component];
         V[d] = u / T;
         u2 += u * u;
      }

      V[density_component] = log(W[density_component] / std::pow(T, 1.0/(gas_gamma-1.0))) 
                           - 0.5 * u2 / T;
      V[energy_component] = -1.0 / T;
   }
   

   // @sect4{EulerEquations::compute_refinement_indicators}
   
   // In this class, we also want to specify
   // how to refine the mesh. The class
   // <code>ConservationLaw</code> that will
   // use all the information we provide
   // here in the <code>EulerEquation</code>
   // class is pretty agnostic about the
   // particular conservation law it solves:
   // as doesn't even really care how many
   // components a solution vector
   // has. Consequently, it can't know what
   // a reasonable refinement indicator
   // would be. On the other hand, here we
   // do, or at least we can come up with a
   // reasonable choice: we simply look at
   // the gradient of the density, and
   // compute
   // $\eta_K=\log\left(1+|\nabla\rho(x_K)|\right)$,
   // where $x_K$ is the center of cell $K$.
   //
   // There are certainly a number of
   // equally reasonable refinement
   // indicators, but this one does, and it
   // is easy to compute:
   static
   void
   compute_refinement_indicators (const dealii::DoFHandler<dim> 		&dof_handler,
                                  const dealii::Mapping<dim>    		&mapping,
                                  const dealii::parallel::distributed::Vector<double>  	&solution,
                                  dealii::Vector<double>		&refinement_indicators) //dealii::TrilinosWrappers::MPI::Vector
   {
      const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
      std::vector<unsigned int> dofs (dofs_per_cell);
      
      const dealii::QMidpoint<dim>  quadrature_formula;
      const dealii::UpdateFlags update_flags = dealii::update_gradients;
      dealii::FEValues<dim> fe_v (mapping, dof_handler.get_fe(),
                                  quadrature_formula, update_flags);
      
      std::vector<std::vector<dealii::Tensor<1,dim> > >
      dU (1, std::vector<dealii::Tensor<1,dim> >(n_components));
      
      typename dealii::DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
      for (unsigned int cell_no=0; cell!=endc; ++cell, ++cell_no)
      if(cell->is_locally_owned())
      {
         fe_v.reinit(cell);
         fe_v.get_function_gradients (solution, dU);

         refinement_indicators(cell_no)
				  = std::log(1+
                    std::sqrt(dU[0][density_component] *
                              dU[0][density_component]));
      }
      //refinement_indicators.compress(VectorOperation::insert);

   }
   
   
   
   // @sect4{EulerEquations::Postprocessor}
   
   // Finally, we declare a class that
   // implements a postprocessing of data
   // components. The problem this class
   // solves is that the variables in the
   // formulation of the Euler equations we
   // use are in conservative rather than
   // physical form: they are momentum
   // densities $\mathbf m=\rho\mathbf v$,
   // density $\rho$, and energy density
   // $E$. What we would like to also put
   // into our output file are velocities
   // $\mathbf v=\frac{\mathbf m}{\rho}$ and
   // pressure $p=(\gamma-1)(E-\frac{1}{2}
   // \rho |\mathbf v|^2)$.
   //
   // In addition, we would like to add the
   // possibility to generate schlieren
   // plots. Schlieren plots are a way to
   // visualize shocks and other sharp
   // interfaces. The word "schlieren" is a
   // German word that may be translated as
   // "striae" -- it may be simpler to
   // explain it by an example, however:
   // schlieren is what you see when you,
   // for example, pour highly concentrated
   // alcohol, or a transparent saline
   // solution, into water; the two have the
   // same color, but they have different
   // refractive indices and so before they
   // are fully mixed light goes through the
   // mixture along bent rays that lead to
   // brightness variations if you look at
   // it. That's "schlieren". A similar
   // effect happens in compressible flow
   // because the refractive index
   // depends on the pressure (and therefore
   // the density) of the gas.
   //
   // The origin of the word refers to
   // two-dimensional projections of a
   // three-dimensional volume (we see a 2d
   // picture of the 3d fluid). In
   // computational fluid dynamics, we can
   // get an idea of this effect by
   // considering what causes it: density
   // variations. Schlieren plots are
   // therefore produced by plotting
   // $s=|\nabla \rho|^2$; obviously, $s$ is
   // large in shocks and at other highly
   // dynamic places. If so desired by the
   // user (by specifying this in the input
   // file), we would like to generate these
   // schlieren plots in addition to the
   // other derived quantities listed above.
   //
   // The implementation of the algorithms
   // to compute derived quantities from the
   // ones that solve our problem, and to
   // output them into data file, rests on
   // the DataPostprocessor class. It has
   // extensive documentation, and other
   // uses of the class can also be found in
   // step-29. We therefore refrain from
   // extensive comments.
   class Postprocessor : public dealii::DataPostprocessor<dim>
   {
   public:
      Postprocessor (const bool do_schlieren_plot);
      
      virtual
      void
      compute_derived_quantities_vector 
         (const std::vector<dealii::Vector<double> >              &uh,
          const std::vector<std::vector<dealii::Tensor<1,dim> > > &duh,
          const std::vector<std::vector<dealii::Tensor<2,dim> > > &dduh,
          const std::vector<dealii::Point<dim> >                  &normals,
          const std::vector<dealii::Point<dim> >                  &evaluation_points,
          std::vector<dealii::Vector<double> >                    &computed_quantities) const;
      
      virtual std::vector<std::string> get_names () const;
      
      virtual
      std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
      get_data_component_interpretation () const;
      
      virtual dealii::UpdateFlags get_needed_update_flags () const;
      
      virtual unsigned int n_output_variables() const;
      
   private:
      const bool do_schlieren_plot;
   };
};

//---------------------------------------------------------------------------
// MHD equations
//---------------------------------------------------------------------------
   

template <int dim>
struct MHDEquations
{
   static const unsigned int v_components = 3;
   // First dim components correspond to momentum
   static const unsigned int n_components             = 2*v_components + 2;
   static const unsigned int density_component        = 2*v_components;
   static const unsigned int energy_component         = 2*v_components+1;
   static const unsigned int momentum_component	      = 0;
   static const unsigned int magnetic_component       = v_components;
   
   static const unsigned int model = 1;
   
   
   
   static
   std::vector<std::string>
   component_names ()
   {
      std::vector<std::string> names;
      names.push_back ("XMomentum");
      names.push_back ("YMomentum");
      names.push_back ("ZMomentum");
      names.push_back ("XMagnetic");
      names.push_back ("YMagnetic");
      names.push_back ("ZMagnetic");
      names.push_back ("Density");
      names.push_back ("Energy");
      
      return names;
   }
   
   
   static
   std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
   component_interpretation ()
   {
      std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation
      (dim, dealii::DataComponentInterpretation::component_is_part_of_vector);
      data_component_interpretation
      .push_back (dealii::DataComponentInterpretation::component_is_scalar);
      data_component_interpretation
      .push_back (dealii::DataComponentInterpretation::component_is_scalar);
      data_component_interpretation
      .push_back (dealii::DataComponentInterpretation::component_is_scalar);
      data_component_interpretation
      .push_back (dealii::DataComponentInterpretation::component_is_scalar);
      data_component_interpretation
      .push_back (dealii::DataComponentInterpretation::component_is_scalar);
      data_component_interpretation
      .push_back (dealii::DataComponentInterpretation::component_is_scalar);
      
      return data_component_interpretation;
   }
   
   // Ratio of specific heats
   static const double gas_gamma;
   
   //---------------------------------------------------------------------------
   // Compute kinetic energy from conserved variables
   //---------------------------------------------------------------------------
   template <typename number, typename InputVector>
   static
   number
   compute_kinetic_energy (const InputVector &W)
   {
      number kinetic_energy = 0;
      // Momentum variables
      for (unsigned int d=0; d<v_components; ++d)
         kinetic_energy += (*(W.begin()+momentum_component+d))*
			   (*(W.begin()+momentum_component+d));
      kinetic_energy *= 0.5/(*(W.begin() + density_component));
      return kinetic_energy;
   }

   //---------------------------------------------------------------------------
   // Compute magnetic pressure from conserved variables
   //---------------------------------------------------------------------------   
   template <typename number, typename InputVector>
   static
   number
   compute_magnetic_pressure (const InputVector &W)
   {
      number magnetic_pressure = 0;
      for (unsigned int d=0; d<v_components; ++d)
         magnetic_pressure += (*(W.begin()+magnetic_component+d))*
			      (*(W.begin()+magnetic_component+d));
      magnetic_pressure*=0.5;
      /*if(isnan(magnetic_pressure))
	std::cout<<"\n\t01="<<W[magnetic_component]<<"\t02="<<W[magnetic_component+1]<<"\t03="<<W[magnetic_component+2];//*/
      return magnetic_pressure;
   }
   
   //---------------------------------------------------------------------------
   // Compute pressure from conserved variables
   //---------------------------------------------------------------------------
   template <typename number, typename InputVector>
   static
   number
   compute_pressure (const InputVector &W)
   {
     number pressure=0;
     number m_pressure = compute_magnetic_pressure<number>(W);
     number kinetic = compute_kinetic_energy<number>(W);
     pressure = ((gas_gamma-1.0)*( *(W.begin() + energy_component)
                - kinetic
		- m_pressure));
     
     return pressure;
   }
   
   //---------------------------------------------------------------------------
   // Speed paremeters (They should be needed just for the numerical fluxes and 
   //			for the CFL condition)
   //---------------------------------------------------------------------------

   //---------------------------------------------------------------------------
   // Compute sound Alfven speed
   //---------------------------------------------------------------------------
   template <typename InputVector>
   static
   typename InputVector::value_type
   alfven_speed (const InputVector &W,
		 const dealii::Tensor<1,dim> &normal)

   {
     typedef typename InputVector::value_type number;
     number B_n = 0;
     for (unsigned int i = 0; i<dim; i++)
       B_n += *(W.begin()+magnetic_component+i)*(normal[i]);
     const number C_a = B_n/std::sqrt(*(W.begin()+density_component));
     /*if(C_a>1e-10)
       std::cout<<"\n\t WTF C_a!=0 ... B_n="<<B_n<<", C_a="<<C_a ;//*/
     return C_a;
   }
   
   // In case we need to compute it over the mean values
   template <typename InputVector>
   static
   typename InputVector::value_type
   alfven_speed (const InputVector &W)

   {
     typedef typename InputVector::value_type number;
     number B_n = std::fabs(*(W.begin()+magnetic_component));
     for (unsigned int i = 1; i<v_components; i++)
       B_n = std::min(B_n, std::fabs(*(W.begin()+magnetic_component+i)));//*/
     const number C_A = B_n/std::sqrt((*(W.begin()+density_component)));
     //std::cout<<"\nAlfven speed: "<<C_A<<"\n";
     return C_A;
   }
   
   //---------------------------------------------------------------------------
   // Compute sound slow acoustic speed
   //---------------------------------------------------------------------------
   template <typename InputVector>
   static
   typename InputVector::value_type
   slow_speed (const InputVector &W,
	       const dealii::Tensor<1,dim> &normal)
   {
      typedef typename InputVector::value_type number;

      const number pressure = compute_pressure<number> (W);
      const number C_a = alfven_speed(W, normal);
      const number a_2 = gas_gamma * pressure / (*(W.begin()+density_component));
      number b_2 = 0;
      for(unsigned int d=0; d<v_components; d++)
	b_2 += (*(W.begin()+magnetic_component+d))* (*(W.begin()+magnetic_component+d));
      b_2 /= (*(W.begin()+density_component));

      number radical1 = (a_2+b_2)*(a_2+b_2) - 4*a_2*C_a*C_a;
      number radical2 =  0.5*(a_2 + b_2 - std::sqrt(radical1));
      number C_s = std::sqrt(radical2);
      /*if(isnan(C_s))
	std::cout<<"\n\t The Slow speed is NaN. r1="<<radical1<<", r2="<<radical2<<", C_a="<<C_a
		 <<", a2="<<a_2<<", b2="<<b_2;//*/
      return C_s;
   }
   
   // In case we need to compute it over the mean values
   template <typename InputVector>
   static
   typename InputVector::value_type
   slow_speed (const InputVector &W)
   {
      typedef typename InputVector::value_type number;

      const number pressure = compute_pressure<number> (W);
      const number C_a = alfven_speed(W);
      if(C_a<1e-12)
	return 0.0;
      const number a_2 = gas_gamma * pressure / (*(W.begin()+density_component));
      number b_2 = 0;
      for(unsigned int d=0; d<v_components; d++)
	b_2 += (*(W.begin()+magnetic_component+d))* (*(W.begin()+magnetic_component+d));
      b_2 /= (*(W.begin()+density_component));

      number radical1 = (a_2+b_2)*(a_2+b_2) - 4*a_2*C_a*C_a;
      number radical2 =  0.5*(a_2 + b_2 - std::sqrt(radical1));
      number C_s = std::sqrt(radical2);
      /*if(isnan(C_s))
	std::cout<<"\n\t The Slow speed is NaN. r1="<<radical1<<", r2="<<radical2<<", C_a="<<C_a
		 <<", a2="<<a_2<<", b2="<<b_2;//*/
      return C_s;
   }
   
   //---------------------------------------------------------------------------
   // Compute sound fast acoustic speed
   //---------------------------------------------------------------------------
   template <typename InputVector>
   static
   typename InputVector::value_type
   fast_speed (const InputVector &W,
	       const dealii::Tensor<1,dim> &normal)

   {
      typedef typename InputVector::value_type number;

      const number pressure = compute_pressure<number> (W);
      const number C_a = alfven_speed(W, normal);

      const number a_2 = gas_gamma * pressure / (*(W.begin()+density_component));
      if(isnan(a_2)) std::cout<<"\n\t a_2 is nan in the fast speed";

      number b_2 = 0;
      for(unsigned int d=0; d<v_components; d++)
	b_2 += (*(W.begin()+magnetic_component+d))* (*(W.begin()+magnetic_component+d));
      b_2 /= (*(W.begin()+density_component));
      if(isnan(b_2)) std::cout<<"\n\t b_2 is nan in the fast speed";

      number radical1 = (a_2+b_2)*(a_2+b_2) - 4*a_2*C_a*C_a;
      number radical2 = 0.5*(a_2 + b_2 + std::sqrt(radical1));
      number C_f = std::sqrt(radical2);
      /*if(isnan(C_f))
	std::cout<<"\n\t The fast speed is NaN. mx="<<*(W.begin())<<", my="<<*(W.begin()+1)<<", e="<<*(W.begin()+energy_component)
		 <<", bx="<<*(W.begin()+dim)<<", by="<<*(W.begin()+1)<<", density="<<*(W.begin()+density_component);//*/
      return C_f;
   }
   
   // In case we need to compute it over the mean values
   template <typename InputVector>
   static
   typename InputVector::value_type
   fast_speed (const InputVector &W)

   {
      typedef typename InputVector::value_type number;

      const number pressure = compute_pressure<number> (W);
      const number C_a = alfven_speed(W);
      const number a_2 = gas_gamma * pressure / (*(W.begin()+density_component));
      number b_2 = 0;
      for(unsigned int d=0; d<v_components; d++)
	b_2 += (*(W.begin()+magnetic_component+d))* (*(W.begin()+magnetic_component+d));
      b_2 /= (*(W.begin()+density_component));

      number C_f = std::sqrt( 0.5*(a_2 + b_2 + std::sqrt((a_2+b_2)*(a_2+b_2) - 4*a_2*C_a*C_a)));
      return C_f;
   }
   
   //---------------------------------------------------------------------------
   // Check speeds orders
   //---------------------------------------------------------------------------
   template <typename InputVector>
   static
   void 
   check_speeds_order (const InputVector &W,
		   const dealii::Tensor<1,dim> &normal)
   {
     typedef typename InputVector::value_type number;
     number C_a=alfven_speed(W,normal);
     number C_s=slow_speed(W,normal);
     number C_f=fast_speed(W,normal);
     if(!((C_a<C_f)||(C_s<C_a)))
       std::cout<<"\n\t (Vectorial) Order of speed invalid! \t C_s: "<<C_s<<", C_a: "<<C_a<<", C_f: "<<C_f<<" \n";
     
   }
   
   //---------------------------------------------------------------------------
   // Check speeds orders
   //---------------------------------------------------------------------------
   template <typename InputVector>
   static
   void 
   check_speeds_order (const InputVector &W)
   {
     typedef typename InputVector::value_type number;
     number C_a=alfven_speed(W);
     number C_s=slow_speed(W);
     number C_f=fast_speed(W);
     if(!((C_a<C_f)||(C_s<C_a)))
       std::cout<<"\n\t (Scalar) Order of speed invalid! \t C_s: "<<C_s<<", C_a: "<<C_a<<", C_f: "<<C_f<<" \n";
     
   }
   
   //---------------------------------------------------------------------------
   // Compute maximum eigenvalue
   //---------------------------------------------------------------------------
   template <typename InputVector>
   static
   typename InputVector::value_type
   max_eigenvalue (const InputVector &W,
		   const dealii::Tensor<1,dim> &normal)
   {
      typedef typename InputVector::value_type number;
      
      const number pressure = compute_pressure<number> (W);
      if(pressure<0)
	std::cout<<"\n\t  pressure is negative at VMaxEigen W=["
		 <<W[0]<<", "<<W[1]<<", "<<W[2]<<", "<<W[3]<<", "
		 <<W[4]<<", "<<W[5]<<", "<<W[6]<<", "<<W[7]<<"]";
      const number sonic = std::sqrt(gas_gamma * pressure / (*(W.begin()+density_component)));

      number velocity = 0;
      for(unsigned int  d=0; d < dim; d++)
	velocity += (*(W.begin()+momentum_component+d)) * normal[d];
      velocity /= (*(W.begin()+density_component)); //std::sqrt(velocity)

      return std::fabs(velocity)+sonic;
   }
   
   // In case we need to compute it over the mean values
   template <typename InputVector>
   static
   typename InputVector::value_type
   max_eigenvalue (const InputVector &W)
   {
     typedef typename InputVector::value_type number;
     const number pressure = compute_pressure<number> (W);
     const number m_pressure = compute_magnetic_pressure<number>(W);
     
     number velocity = 0;
     for (unsigned int d=0; d<dim; ++d)
       velocity += *(W.begin()+d) *
		   *(W.begin()+d);
     velocity = std::sqrt(velocity) / (*(W.begin()+density_component));
     
     return velocity + std::sqrt((gas_gamma * pressure + 2*m_pressure) / (*(W.begin()+density_component)));
   }
   
   //---------------------------------------------------------------------------
   // Compute maximum eigenvalue in normal direction
   //---------------------------------------------------------------------------
   template <typename InputVector>
   static
   typename InputVector::value_type
   max_eigenvalue_normal (const InputVector        &W,
                   const dealii::Tensor<1,dim> &normal)
   {
     typedef typename InputVector::value_type number;
     const number pressure = compute_pressure<number> (W);
     const number m_pressure = compute_magnetic_pressure<number>(W);
     
     const number sonic = std::sqrt((gas_gamma * pressure + 2*m_pressure) / (*(W.begin()+density_component)));
     
     number velocity = 0;
     for (unsigned int d=0; d<dim; ++d)
       velocity += *(W.begin()+d) * normal[d];
     
     velocity /=  (*(W.begin()+density_component));
     return std::fabs(velocity) + sonic;
      
   }
   
   // In case we need to compute it over the mean values
   template <typename InputVector>
   static
   typename InputVector::value_type
   sound_speed (const InputVector &W)
   {
     typedef typename InputVector::value_type number;
     const number pressure = compute_pressure<number> (W);
     const number m_pressure = compute_magnetic_pressure<number>(W);
     return std::sqrt((gas_gamma * pressure + 2*m_pressure) / (*(W.begin()+density_component)));
     
   }
   
   //---------------------------------------------------------------------------
   // Compute cartesian components of flux
   //---------------------------------------------------------------------------
   template <typename InputVector, typename number>
   static
   void compute_flux_matrix (const InputVector &W,
                             number (&flux)[n_components][dim])
   {
      // First compute the pressure that
      // appears in the flux matrix, and
      // then compute the first
      // <code>dim</code> columns of the
      // matrix that correspond to the
      // momentum terms:
      const number pressure = compute_pressure<number> (W);
      const number magnetic_pressure = compute_magnetic_pressure<number> (W);
      //const number density_1=1.0/W[density_component];
      
      // Compute the flux function for the momentum
      for (unsigned int i=0; i<v_components; ++i)
      {
	for (unsigned int j=0; j<dim; ++j)
	{
            flux[momentum_component+i][j] = W[momentum_component+i]*W[momentum_component+j]/W[density_component]
					    - W[magnetic_component+i]*W[magnetic_component+j];
	if(j==i)
	  flux[momentum_component+i][j] += pressure + magnetic_pressure;
	
	if(isnan(flux[momentum_component+i][j]))
	  std::cout<<"\n\t flux function is NaN, Pressure="<<pressure<<", ma_press="<<magnetic_pressure
		   <<", moment="<<W[momentum_component+i]<<", density_1="<<W[density_component];//*/
	}
      }
      
      // Compute the flux function for the magnetic field
	for (unsigned int i=0; i<v_components; ++i)
	{
	  for (unsigned int j=0; j<dim; ++j)
	  {
	    flux[magnetic_component+i][j] = (W[momentum_component+j] * W[magnetic_component+i]
					    - W[momentum_component+i] * W[magnetic_component+j])
					    / W[density_component];
	    if(j==i)
	      flux[magnetic_component+i][j] = 0;
	    
	    if(isnan(flux[magnetic_component+i][j]))
	      std::cout<<"\n\t flux function is NaN, moment="<<W[momentum_component+i]
		       <<", mag="<<W[magnetic_component+i]<<", density_1="<<W[density_component];//*/
	  }
	}
      
      // Flux function for the conservation of mass
      for (unsigned int i=0; i<dim; ++i)
	flux[density_component][i] = W[momentum_component+i];
      
      number udotB = 0;
      for (unsigned int i=0; i<v_components;++i)
	udotB += W[momentum_component+i] * W[magnetic_component+i];
      udotB/=W[density_component];

      // Flux function for the conservation of energy
      for (unsigned int i=0; i<dim; ++i)
         flux[energy_component][i] = W[momentum_component+i] * ( W[energy_component]
				     + pressure + magnetic_pressure) / W[density_component]
                                     - W[magnetic_component+i]*udotB;
   }
   
   //---------------------------------------------------------------------------
   // Compute Powell Terms
   //---------------------------------------------------------------------------
   template <typename InputVector, typename number>
   static
   void powell_terms (const InputVector &W,
                          number (&flux)[n_components])
   {
     number density_1=1/W[density_component];
     number velocity;
     
     flux[density_component] = 0;
     flux[energy_component] = 0;
     for (unsigned int i=0; i<v_components; ++i)
     {
       velocity = W[momentum_component+i]*density_1;
       
       // Powell terms for the momentum components
       flux[momentum_component + i] = W[magnetic_component + i];
       // Powell terms for the magnetic components
       flux[magnetic_component + i] = velocity;
       // Powell terms for the energy component
       flux[energy_component] += velocity*W[magnetic_component + i];
     }
     
   }
   
   //---------------------------------------------------------------------------
   // Compute flux along normal
   //---------------------------------------------------------------------------
   /*template <typename InputVector, typename number>
   static
   void normal_flux (const InputVector        	 &W,
                     const dealii::Tensor<1,dim> &normal,
                     number                   	(&flux)[n_components])
   {
      const number pressure = compute_pressure<number> (W);
      
      number vdotn = 0.0;
      for (unsigned int d=0; d<dim; ++d)
         vdotn += W[d] * normal[d];
      vdotn /= W[density_component];
      
      flux[density_component] = W[density_component] * vdotn;
      flux[energy_component] = (W[energy_component] + pressure) * vdotn;
      for (unsigned int d=0; d<dim; ++d)
         flux[d] = pressure * normal[d] + W[d] * vdotn;
   }//*/
   
   //---------------------------------------------------------------------------
   // Left and right eigenvector matrices
   // Lx, Rx = along x direction
   // Ly, Ry = along y direction
   // Expressions taken from
   // http://people.nas.nasa.gov/~pulliam/Classes/New_notes/euler_notes.pdf
   // Note: This is implemented only for 2-D
   //---------------------------------------------------------------------------
   /*static
   void compute_eigen_matrix (const dealii::Vector<double> &W,
                              double            (&Rx)[n_components][n_components],
                              double            (&Lx)[n_components][n_components],
                              double            (&Ry)[n_components][n_components],
                              double            (&Ly)[n_components][n_components])
   {
      double g1   = gas_gamma - 1.0;
      double rho  = W[density_component];
      double E    = W[energy_component];
      double u    = W[0] / rho;
      double v    = W[1] / rho;
      double q2   = u*u + v*v;
      double p    = g1 * (E - 0.5 * rho * q2);
      double c2   = gas_gamma * p / rho;
      double c    = std::sqrt(c2);
      double beta = 0.5/c2;
      double phi2 = 0.5*g1*q2;
      double h    = c2/g1 + 0.5*q2;
      
      Rx[0][0] = 1;      Rx[0][1] = 0;  Rx[0][2] = 1;     Rx[0][3] = 1;
      Rx[1][0] = u;      Rx[1][1] = 0;  Rx[1][2] = u+c;   Rx[1][3] = u-c;
      Rx[2][0] = v;      Rx[2][1] = -1; Rx[2][2] = v;     Rx[2][3] = v;
      Rx[3][0] = 0.5*q2; Rx[3][1] = -v; Rx[3][2] = h+c*u; Rx[3][3] = h-c*u;
      
      Ry[0][0] = 1;      Ry[0][1] = 0;  Ry[0][2] = 1;     Ry[0][3] = 1;
      Ry[1][0] = u;      Ry[1][1] = 1;  Ry[1][2] = u;     Ry[1][3] = u;
      Ry[2][0] = v;      Ry[2][1] = 0;  Ry[2][2] = v+c;   Ry[2][3] = v-c;
      Ry[3][0] = 0.5*q2; Ry[3][1] = u;  Ry[3][2] = h+c*v; Ry[3][3] = h-c*v;
      
      Lx[0][0] = 1-phi2/c2;       Lx[0][1] = g1*u/c2;       Lx[0][2] = g1*v/c2;    Lx[0][3] = -g1/c2;
      Lx[1][0] = v;               Lx[1][1] = 0;             Lx[1][2] = -1;         Lx[1][3] = 0;
      Lx[2][0] = beta*(phi2-c*u); Lx[2][1] = beta*(c-g1*u); Lx[2][2] = -beta*g1*v; Lx[2][3] = beta*g1;
      Lx[3][0] = beta*(phi2+c*u); Lx[3][1] =-beta*(c+g1*u); Lx[3][2] = -beta*g1*v; Lx[3][3] = beta*g1;
      
      Ly[0][0] = 1-phi2/c2;       Ly[0][1] = g1*u/c2;       Ly[0][2] = g1*v/c2;       Ly[0][3] = -g1/c2;
      Ly[1][0] = -u;              Ly[1][1] = 1;             Ly[1][2] = 0;             Ly[1][3] = 0;
      Ly[2][0] = beta*(phi2-c*v); Ly[2][1] =-beta*g1*u;     Ly[2][2] = beta*(c-g1*v); Ly[2][3] = beta*g1;
      Ly[3][0] = beta*(phi2+c*v); Ly[3][1] =-beta*g1*u;     Ly[3][2] =-beta*(c+g1*v); Ly[3][3] = beta*g1;
      
   }//*/
   
   //---------------------------------------------------------------------------
   // convert from conserved to characteristic variables: W = L*W
   //---------------------------------------------------------------------------
   /*static
   void transform_to_char (const double           (&L)[n_components][n_components],
                           dealii::Vector<double> &W)
   {
      dealii::Vector<double> V(n_components);
      
      V[0] = W[density_component];
      V[n_components-1] = W[energy_component];
      for(unsigned int d=0; d<dim; ++d)
         V[d+1] = W[d];
      
      W = 0;
      for(unsigned int i=0; i<n_components; ++i)
         for(unsigned int j=0; j<n_components; ++j)
            W[i] += L[i][j] * V[j];
   }//*/
   
   //---------------------------------------------------------------------------
   // convert from characteristic to conserved variables: W = R*W
   //---------------------------------------------------------------------------
   /*static
   void transform_to_con (const double           (&R)[n_components][n_components],
                          dealii::Vector<double> &W)
   {
      dealii::Vector<double> V(n_components);
      
      V = 0;
      for(unsigned int i=0; i<n_components; ++i)
         for(unsigned int j=0; j<n_components; ++j)
            V[i] += R[i][j] * W[j];

      W[density_component] = V[0];
      W[energy_component] = V[n_components-1];
      for(unsigned int d=0; d<dim; ++d)
         W[d] = V[d+1];
      
   }//*/

   // @sect4{EulerEquations::compute_normal_flux}
   
   // On the boundaries of the
   // domain and across hanging
   // nodes we use a numerical flux
   // function to enforce boundary
   // conditions.  This routine is
   // the basic Lax-Friedrich's flux
   // with a stabilization parameter
   // $\alpha$. It's form has also
   // been given already in the
   // introduction:

   // --------------------------------------------------------------------------
   // Local lax-Friedrichs flux
   // --------------------------------------------------------------------------
   template <typename InputVector>
   static
   void lxf_flux 
   (
    const dealii::Tensor<1,dim> &normal,
    const InputVector                &Wplus,
    const InputVector                &Wminus,
    const dealii::Vector<double>     &Aplus,
    const dealii::Vector<double>     &Aminus,
    typename InputVector::value_type (&normal_flux)[n_components]
   )
   {
      typedef typename InputVector::value_type number;
      
      number fluxplus[n_components][dim], fluxminus[n_components][dim];
      compute_flux_matrix (Wplus, fluxplus);
      compute_flux_matrix (Wminus, fluxminus);
      
      for(unsigned int i=0; i<n_components;i++)
      {
	number i_component=0;
	for(unsigned int j=0; j<dim; j++)
	  i_component+=0.5*(fluxplus[i][j]+fluxminus[i][j])*normal[j];
	normal_flux[i]=i_component;
      }
      
      // Maximum eigenvalue at cell face
      number lambda_plus = max_eigenvalue_normal(Aplus, normal);
      number lambda_minus = max_eigenvalue_normal(Aminus, normal);
      number lambda = std::max(lambda_plus, lambda_minus);
      //number lambda=1;
      if(isnan(lambda)||isnan(lambda_minus)||isnan(lambda_plus))
	std::cout<<"\n \t Numerical flux problem lambda = "<<lambda<<"\t lambda_minus = "<<lambda_minus
		 << "\t lambda_plus = "<< lambda_plus;
      
      // Dissipation flux
      
      for (unsigned int c=0; c<n_components; ++c)
         normal_flux[c] += 0.5 * lambda  * (Wplus[c] - Wminus[c]);//*/
	 
      /*typedef typename InputVector::value_type number;

      // Normal velocity
      number vdotn_plus=0, vdotn_minus=0, bdotn_plus=0, bdotn_minus=0;
      
      for(unsigned int d=0; d<dim; ++d)
      {
         vdotn_plus  += Wplus[d]  * normal[d];
         vdotn_minus += Wminus[d] * normal[d];
	 bdotn_plus  += Wplus[magnetic_component+d]  * normal[d];
         bdotn_minus += Wminus[magnetic_component+d] * normal[d];
      }
      
      vdotn_plus  /= Wplus [density_component];
      vdotn_minus /= Wminus[density_component];
      
      // pressure
      number p_plus, p_minus, mp_plus, mp_minus;

      p_plus  = compute_pressure<number> (Wplus);
      mp_plus = compute_magnetic_pressure<number> (Wplus);
      p_minus = compute_pressure<number> (Wminus);
      mp_minus = compute_magnetic_pressure<number> (Wminus);
      
      // Maximum eigenvalue at cell face
      number lambda_plus = max_eigenvalue_normal (Aplus, normal);
      number lambda_minus = max_eigenvalue_normal (Aminus, normal);
      number lambda = std::max(lambda_plus, lambda_minus);
      if(isnan(lambda))
 	std::cout<<"\n\t Lambda NaN";
      
      // Momentum flux
      number normal_p[v_components];
      for(unsigned int d=0;d<dim;d++)
	normal_p[d] =normal[d];
      normal_p[dim]=0;

      for (unsigned int i=0; i<v_components; ++i)
      {
            normal_flux[i] = 0.5 * ( (p_plus+mp_plus)*normal_p[i] + Wplus[i]*vdotn_plus
				- Wplus[magnetic_component+i]*bdotn_plus +
				(p_minus+mp_minus) * normal_p[i] + Wminus[i]*vdotn_minus
				- Wminus[magnetic_component+i]*bdotn_minus);
      }
      
      // Magnetic flux
      for (unsigned int i=0; i<v_components; ++i)
      {
	normal_flux[magnetic_component+i] = 0.5*(vdotn_plus * Wplus[magnetic_component+i] -
					    Wplus[momentum_component+i] * bdotn_plus/Wplus[density_component] +
					    vdotn_minus * Wminus[magnetic_component+i]) -
					    Wminus[momentum_component+i] * bdotn_minus/Wminus[density_component];
      }

      // Density flux
      normal_flux[density_component] = 0.5 * (Wplus [density_component] * vdotn_plus +
                                              Wminus[density_component] * vdotn_minus);
      
      // Energy flux
      number udotBplus = 0, udotBminus=0;
      for (unsigned int i=0; i<v_components;++i)
      {
	udotBplus += Wplus[momentum_component+i] * Wplus[magnetic_component+i];
	udotBminus += Wminus[momentum_component+i] * Wminus[magnetic_component+i];
      }
      udotBplus/=Wplus[density_component];
      udotBminus/=Wminus[density_component];
      
      normal_flux[energy_component] = 0.5*(vdotn_plus*(Wplus[energy_component] + p_plus + mp_plus)
                                      - bdotn_plus*udotBplus
				      + vdotn_minus*(Wminus[energy_component] + p_minus + mp_minus)
                                      - bdotn_minus*udotBminus);
      
      
      // Dissipation flux
      for (unsigned int c=0; c<n_components; ++c)
         normal_flux[c] += 0.5 * lambda * (Wplus[c] - Wminus[c]);//*/
   }
   
   template <typename Number>
   static Number logavg(Number  left, Number right)
   {
     Number roeaverage=0;
     Number F = 0;
     Number psi = left/right;
     Number f   = (psi-1)/(psi+1);
     Number u   = f * f;
     
     if(u<1e-2)
       F = 1 + u/3 + u*u/5 + u*u*u/7;
     else
       F = 0.5*log(psi)/f;
     
     roeaverage = (left+right)/(2*F);
     
     return roeaverage;
   }
   
   
   /*template <typename InputVector>
   static void computez(typename InputVector::value_type (&R)[n_components][n_components],
			//const    InputVector              &prim,
			typename InputVector::value_type (&prim)[n_components],
			typename InputVector::value_type (&z)[n_components])//*/
   template <typename number>
   static void computez(number (&R)[n_components][n_components],
			number (&prim)[n_components],
			number (&z)[n_components])
   {
     //typedef typename InputVector::value_type number;
     //number p    = compute_pressure<number>(prim),
     number s    = log(prim[4]) - gas_gamma * log(prim[0]),
	    beta = 0.5*prim[0]/prim[4],
	    q2   = (prim[1]*prim[1] + prim[2]*prim[2] + prim[3]*prim[3]),
	    q[n_components];
     
     // Compute entropy variables
     q[0] = -s/(gas_gamma-1.0) - beta*q2;
     q[1] = 2.0*beta*prim[1];
     q[2] = 2.0*beta*prim[2];
     q[3] = 2.0*beta*prim[3];
     q[4] = -2.0*beta;
     q[5] = 2.0*beta*prim[5];
     q[6] = 2.0*beta*prim[6];
     q[7] = 2.0*beta*prim[7];
     
     // Compute R^T * q
     for(unsigned int i=0; i<n_components;i++)
     {
       z[i] = 0.0;
       for(unsigned int j=0; j<n_components;j++)
	 z[i] += R[j][i] * q[j];
     }
   
   }
   
   
   // --------------------------------------------------------------------------
   // Entropy stable numerical flux by Chandrashekar and Klingenberg
   // --------------------------------------------------------------------------
   template <typename InputVector>
   static
   void es_flux 
   (
    const dealii::Tensor<1,dim> &normal,
    const InputVector                &Wplus,
    const InputVector                &Wminus,
    typename InputVector::value_type (&normal_flux)[n_components]
   )
   {
     typedef typename InputVector::value_type number;
     
     number left[n_components], right[n_components], flux[n_components];
     number rho, u[v_components], ul2, ur2, q2, B1, B2, B3, Bl2, Br2, mB2;
     number betal, betar, beta, rho_a, beta_a, p, bu1, bu2, bu3;
     number a, srho, bb1, bb2, bb3, mbb2, cf2, cs2, cf, cs, alpf, alps;
     number n1, n2, n3, np1, np2, np3, bet, alp, npn1, npn2, npn3, ff;
     number bn, bnp, s1, s2, s3, sbn, t1, t2, t3;
     number Kin, Jac[n_components][n_components], Rp[n_components][n_components],
	    R[n_components][n_components], Lambda[n_components];
     number Diff, zl[n_components], zr[n_components], dz[n_components];
     number unorm, Bnorm, bunorm;
     
     number g_1 = gas_gamma-1, g1 = sqrt(g_1/gas_gamma), g2 = 1/sqrt(gas_gamma),
	    g3 = g2/sqrt(2), g4=1/g_1;
	    
     left[0]=Wplus[6];
     left[1]=Wplus[0]/Wplus[6];
     left[2]=Wplus[1]/Wplus[6];
     left[3]=Wplus[2]/Wplus[6];
     left[4]=compute_pressure<number>(Wplus);
     left[5]=Wplus[3];
     left[6]=Wplus[4];
     left[7]=Wplus[5];
     
     right[0]=Wminus[6];
     right[1]=Wminus[0]/Wminus[6];
     right[2]=Wminus[1]/Wminus[6];
     right[3]=Wminus[2]/Wminus[6];
     right[4]=compute_pressure<number>(Wminus);
     right[5]=Wminus[3];
     right[6]=Wminus[4];
     right[7]=Wminus[5];
     
     rho = logavg (left[0], right[0]);
     u[0]   = 0.5 * (left[1] + right[1]);
     u[1]   = 0.5 * (left[2] + right[2]);
     u[2]   = 0.5 * (left[3] + right[3]);
     
     unorm= u[0]*normal[0] + u[1]*normal[1];
     
     ul2 = left[1]*left[1] + left[2]*left[2] + left[3]*left[3];
     ur2 = right[1]*right[1] + right[2]*right[2] + right[3]*right[3];
     q2  = 0.5 * (ul2 + ur2);

   B1   = 0.5 * (left[5] + right[5]);
   B2   = 0.5 * (left[6] + right[6]);
   B3   = 0.5 * (left[7] + right[7]);
   Bnorm= B1*normal[0] + B2*normal[1];

   Bl2 = left[5]*left[5] + left[6]*left[6] + left[7]*left[7];
   Br2 = right[5]*right[5] + right[6]*right[6] + right[7]*right[7];
   mB2  = 0.5*(Bl2 + Br2);
   
   betal = left[0]/(2.0*left[4]);
   betar = right[0]/(2.0*right[4]);
   beta  = logavg(betal, betar);
   
   rho_a = 0.5*(left[0]+right[0]);
   beta_a = 0.5*(betal+betar);
   p   = 0.5 * rho_a / beta_a;

   bu1   = (betal*left[1] + betar*right[1])/(betal+betar);
   bu2   = (betal*left[2] + betar*right[2])/(betal+betar);
   bu3   = (betal*left[3] + betar*right[3])/(betal+betar);
   bunorm= bu1*normal[0] + bu2*normal[1];
   
   flux[0] = rho * unorm;
   flux[1] = (p + 0.5*mB2)*normal[0] + u[0] * flux[0] - Bnorm * B1;
   flux[2] = (p + 0.5*mB2)*normal[1] + u[1] * flux[0] - Bnorm * B2;
   flux[3] =                           u[2] * flux[0] - Bnorm * B3;
   flux[5] = bunorm * B1 - bu1 * Bnorm;
   flux[6] = bunorm * B2 - bu2 * Bnorm;
   flux[7] = bunorm * B3 - bu3 * Bnorm;
   flux[4] = 0.5*( 1/(g_1*beta) - q2) * flux[0]
             + u[0] * flux[1] + u[1] * flux[2] + u[2] * flux[3]
             + B1 * flux[5] + B2*flux[6] + B3 * flux[7]
             - 0.5*unorm*mB2 + (u[0]*B1+u[1]*B2+u[2]*B3)*Bnorm;
 
   // Add entropy dissipation
   // normal vector
   n1 = normal[0];
   n2 = normal[1];
   n3 = 0.0;

   // Add entropy dissipation
   a = sqrt(0.5 * gas_gamma / beta);
   
   srho = sqrt(rho);
   bb1 = B1/srho;
   bb2 = B2/srho;
   bb3 = B3/srho;
   mbb2 = bb1*bb1 + bb2*bb2 + bb3*bb3;
   bn  = bb1*n1  + bb2*n2  + bb3*n3;
   cf2 = 0.5*(a*a + mbb2) + 0.5*sqrt( (a*a+mbb2)*(a*a+mbb2) - 4.0*a*a*bn*bn);
   cs2 = 0.5*(a*a + mbb2) - 0.5*sqrt( (a*a+mbb2)*(a*a+mbb2) - 4.0*a*a*bn*bn);
   cf = sqrt(cf2);
   cs = sqrt(abs(cs2)); // cs may be zero or close to zero
   
   alpf = sqrt( abs((a*a - cs2)/(cf2 - cs2)) );
   alps = sqrt( abs((cf2 - a*a)/(cf2 - cs2)) );
   
   // vector nperp
   ff = abs(mbb2 - bn*bn);
   if(ff < 1.0e-10)
   {
      if(abs(normal[1]) < 1.0e-14)
      {
         np1 = 0.0;
         np2 = 1.0/sqrt(2.0);
         np3 = 1.0/sqrt(2.0);
      }
      else
      {
         np1 = 1.0/sqrt(2.0);
         np2 = 0.0;
         np3 = 1.0/sqrt(2.0);
      }
  }
   else
   {
      bet = 1.0/sqrt(ff);
      alp = - bet * bn;
      np1 = alp*n1 + bet*bb1;
      np2 = alp*n2 + bet*bb2;
      np3 = alp*n3 + bet*bb3;
   }
   
   // Primitive eigenvectors
   
   // entropy wave
   Rp[0][0] = g1*srho;
   Rp[1][0] = 0;
   Rp[2][0] = 0;
   Rp[3][0] = 0;
   Rp[4][0] = 0;
   Rp[5][0] = 0;
   Rp[6][0] = 0;
   Rp[7][0] = 0;
   
   // divergence wave
   Rp[0][1] = 0;
   Rp[1][1] = 0;
   Rp[2][1] = 0;
   Rp[3][1] = 0;
   Rp[4][1] = 0;
   Rp[5][1] = g2*a*n1;
   Rp[6][1] = g2*a*n2;
   Rp[7][1] = g2*a*n3;
   
   // alfven waves
   // np x n
   npn1 = np2*n3 - np3*n2;
   npn2 = np3*n1 - np1*n3;
   npn3 = np1*n2 - np2*n1;

   Rp[0][2] = 0;
   Rp[1][2] = g3*a*npn1/srho;
   Rp[2][2] = g3*a*npn2/srho;
   Rp[3][2] = g3*a*npn3/srho;
   Rp[4][2] = 0;
   Rp[5][2] = g3*a*npn1;
   Rp[6][2] = g3*a*npn2;
   Rp[7][2] = g3*a*npn3;

   Rp[0][3] =  Rp[0][2];
   Rp[1][3] = -Rp[1][2];
   Rp[2][3] = -Rp[2][2];
   Rp[3][3] = -Rp[3][2];
   Rp[4][3] =  Rp[4][2];
   Rp[5][3] =  Rp[5][2];
   Rp[6][3] =  Rp[6][2];
   Rp[7][3] =  Rp[7][2];

   // fast magneto acoustic wave
   bnp = bb1*np1 + bb2*np2 + bb3*np3;
   s1  = (alpf*a*a*n1 + alps*a*(bnp*n1 - bn*np1))/(srho*cf);
   s2  = (alpf*a*a*n2 + alps*a*(bnp*n2 - bn*np2))/(srho*cf);
   s3  = (alpf*a*a*n3 + alps*a*(bnp*n3 - bn*np3))/(srho*cf);
   
   Rp[0][4] =  g3*alpf*srho;
   Rp[1][4] = -g3*s1;
   Rp[2][4] = -g3*s2;
   Rp[3][4] = -g3*s3;
   Rp[4][4] =  g3*alpf*srho*a*a;
   Rp[5][4] =  g3*alps*a*np1;
   Rp[6][4] =  g3*alps*a*np2;
   Rp[7][4] =  g3*alps*a*np3;
   
   Rp[0][5] =  Rp[0][4];
   Rp[1][5] = -Rp[1][4];
   Rp[2][5] = -Rp[2][4];
   Rp[3][5] = -Rp[3][4];
   Rp[4][5] =  Rp[4][4];
   Rp[5][5] =  Rp[5][4];
   Rp[6][5] =  Rp[6][4];
   Rp[7][5] =  Rp[7][4];
   
   // slow magneto acoustic waves
   if(bn > 0)
   {
      sbn = +1.0;
   }
   else
   {
      sbn = -1.0;
   }
   t1 = sbn*(alps*a*bn*n1 + alpf*cf*cf*np1)/(srho*cf);
   t2 = sbn*(alps*a*bn*n2 + alpf*cf*cf*np2)/(srho*cf);
   t3 = sbn*(alps*a*bn*n3 + alpf*cf*cf*np3)/(srho*cf);
   
   Rp[0][6] =  g3*alps*srho;
   Rp[1][6] = -g3*t1;
   Rp[2][6] = -g3*t2;
   Rp[3][6] = -g3*t3;
   Rp[4][6] =  g3*alps*srho*a*a;
   Rp[5][6] = -g3*alpf*a*np1;
   Rp[6][6] = -g3*alpf*a*np2;
   Rp[7][6] = -g3*alpf*a*np3;

   Rp[0][7] =  Rp[0][6];
   Rp[1][7] = -Rp[1][6];
   Rp[2][7] = -Rp[2][6];
   Rp[3][7] = -Rp[3][6];
   Rp[4][7] =  Rp[4][6];
   Rp[5][7] =  Rp[5][6];
   Rp[6][7] =  Rp[6][6];
   Rp[7][7] =  Rp[7][6];
   
   // Jacobian = d(con)/d(prim)
   Kin = 0.5*(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);

   for(unsigned int i = 0; i < n_components; i++)
   	for(unsigned int j = 0; j < n_components; j++)
	   Jac[i][j]    = 0.0;

   Jac[0][0] = 1.0;
   Jac[1][0] = u[0];
   Jac[2][0] = u[1];
   Jac[3][0] = u[2];
   Jac[4][0] = Kin;

   Jac[1][1] = rho;
   Jac[4][1] = rho*u[0];

   Jac[2][2] = rho;
   Jac[4][2] = rho*u[1];

   Jac[3][3] = rho;
   Jac[4][3] = rho*u[2];

   Jac[4][4] = g4;

   Jac[4][5] = B1;
   Jac[4][6] = B2;
   Jac[4][7] = B3;

   Jac[5][5] = 1.0;
   Jac[6][6] = 1.0;
   Jac[7][7] = 1.0;

   // Conserved eigenvectors: R = Jac * Rp
   for(unsigned int i=0; i<n_components; i++)
   {
      for(unsigned int j=0; j<n_components; j++)
      {
         R[i][j] = 0.0;
         for(unsigned int k=0; k<n_components; k++)
            R[i][j] += Jac[i][k]*Rp[k][j];
      }
   }
   
   Lambda[0] = abs(unorm);
   Lambda[1] = abs(unorm);
   Lambda[2] = abs(unorm-bn);
   Lambda[3] = abs(unorm+bn);
   Lambda[4] = abs(unorm-cf);
   Lambda[5] = abs(unorm+cf);
   Lambda[6] = abs(unorm-cs);
   Lambda[7] = abs(unorm+cs);

   // Compute z
   computez(R, left,  zl);
   computez(R, right, zr);

   for(unsigned int i=0; i<n_components; i++)
      dz[i] = Lambda[i] * (zr[i] - zl[i]);
   
   for(unsigned int i=0; i<n_components; i++)
   {
      Diff = 0.0;
      for(unsigned int j=0; j<n_components; j++)
         Diff += R[i][j] * dz[j];
      
      flux[i] -= 0.5*Diff;
   }
   
   normal_flux[0] = flux[1];
   normal_flux[1] = flux[2];
   normal_flux[2] = flux[3];
   normal_flux[3] = flux[5];
   normal_flux[4] = flux[6];
   normal_flux[5] = flux[7];
   normal_flux[6] = flux[0];
   normal_flux[7] = flux[4];
   
   }
   
   
   
   // --------------------------------------------------------------------------
   // Steger-Warming flux
   // --------------------------------------------------------------------------
   template <typename InputVector>
   static
   void steger_warming_flux 
   (
    const dealii::Point<dim>         &normal,
    const InputVector                &Wplus,
    const InputVector                &Wminus,
    typename InputVector::value_type (&normal_flux)[n_components]
   )
   {
      typedef typename InputVector::value_type number;

      number pflux[n_components], mflux[n_components];
      
      // normal velocity and velocity magnitude
      number vdotn_plus=0, vdotn_minus=0, q2_plus=0, q2_minus=0;

      for(unsigned int d=0; d<dim; ++d)
      {
         vdotn_plus  += Wplus[d]  * normal[d];
         vdotn_minus += Wminus[d] * normal[d];
         
         q2_plus  += Wplus[d]  * Wplus[d];
         q2_minus += Wminus[d] * Wminus[d];
      }
      
      vdotn_plus  /= Wplus [density_component];
      vdotn_minus /= Wminus[density_component];
      
      q2_plus  /= Wplus [density_component] * Wplus [density_component];
      q2_minus /= Wminus[density_component] * Wminus[density_component];
      
      // pressure
      number p_plus, p_minus;
      
      p_plus  = compute_pressure<number> (Wplus);
      p_minus = compute_pressure<number> (Wminus);
      
      // sound speed
      number c_plus, c_minus;
      c_plus  = std::sqrt(gas_gamma * p_plus  / Wplus [density_component]);
      c_minus = std::sqrt(gas_gamma * p_minus / Wminus[density_component]);

      // positive flux
      number l1p, l2p, l3p, ap, fp;
      
      l1p = std::max( vdotn_plus,          0.0);
      l2p = std::max( vdotn_plus + c_plus, 0.0);
      l3p = std::max( vdotn_plus - c_plus, 0.0);
      ap  = 2.0 * (gas_gamma - 1.0) * l1p + l2p + l3p;
      fp  = 0.5 * Wplus[density_component] / gas_gamma;
      
      for(unsigned int d=0; d<dim; ++d)
         pflux[d] = ap * Wplus[d]/Wplus[density_component] +
                          c_plus * (l2p - l3p) * normal[d];
      
      pflux[density_component] = ap;
      pflux[energy_component] = 0.5 * ap * q2_plus +
                                c_plus * vdotn_plus * (l2p - l3p) +
                                c_plus * c_plus * (l2p + l3p) / (gas_gamma - 1.0);
      
      // negative flux
      number l1m, l2m, l3m, am, fm;
      
      l1m = std::min( vdotn_minus,           0.0);
      l2m = std::min( vdotn_minus + c_minus, 0.0);
      l3m = std::min( vdotn_minus - c_minus, 0.0);
      am  = 2.0 * (gas_gamma - 1.0) * l1m + l2m + l3m;
      fm  = 0.5 * Wminus[density_component] / gas_gamma;
      
      for(unsigned int d=0; d<dim; ++d)
         mflux[d] = am * Wminus[d]/Wminus[density_component] +
                    c_minus * (l2m - l3m) * normal[d];
      
      mflux[density_component] = am;
      mflux[energy_component] = 0.5 * am * q2_minus +
                                c_minus * vdotn_minus * (l2m - l3m) +
                                c_minus * c_minus * (l2m + l3m) / (gas_gamma - 1.0);
            
      // Total flux
      for (unsigned int c=0; c<n_components; ++c)
         normal_flux[c] = fp * pflux[c] + fm * mflux[c];
   }
   
   // --------------------------------------------------------------------------
   // Roe flux
   // --------------------------------------------------------------------------
   template <typename InputVector>
   static
   void roe_flux
   (
    const dealii::Point<dim>         &normal,
    const InputVector                &W_l,
    const InputVector                &W_r,
    typename InputVector::value_type (&normal_flux)[n_components]
    )
   {
      typedef typename InputVector::value_type number;
      
      number rho_l_sqrt = std::sqrt(W_l[density_component]);
      number rho_r_sqrt = std::sqrt(W_r[density_component]);
      number fact_l = rho_l_sqrt / (rho_l_sqrt + rho_r_sqrt);
      number fact_r = 1.0 - fact_l;
      
      number v_l[dim], v_r[dim], velocity[dim], dv[dim];
      number v2_l = 0, v2_r = 0;
      number v_l_normal = 0, v_r_normal = 0;
      number vel_normal = 0, v2 = 0;
      number v_dot_dv = 0;
      for(unsigned int d=0; d<dim; ++d)
      {
         v_l[d]      = W_l[d] / W_l[density_component];
         v_r[d]      = W_r[d] / W_r[density_component];
         v2_l       += v_l[d] * v_l[d];
         v2_r       += v_r[d] * v_r[d];
         v_l_normal += v_l[d] * normal[d];
         v_r_normal += v_r[d] * normal[d];
         
         velocity[d] = v_l[d] * fact_l + v_r[d] * fact_r;
         vel_normal += velocity[d] * normal[d];
         v2         += velocity[d] * velocity[d];
         dv[d]       = v_r[d] - v_l[d];
         v_dot_dv   += velocity[d] * dv[d];
      }
      
      number p_l = (gas_gamma-1) * (W_l[energy_component] - 0.5 * W_l[density_component] * v2_l);
      number p_r = (gas_gamma-1) * (W_r[energy_component] - 0.5 * W_r[density_component] * v2_r);
      
      number h_l = gas_gamma * p_l / W_l[density_component] / (gas_gamma-1) + 0.5 * v2_l;
      number h_r = gas_gamma * p_r / W_r[density_component] / (gas_gamma-1) + 0.5 * v2_r;
      
      number density = rho_l_sqrt * rho_r_sqrt;
      number h = h_l * fact_l + h_r * fact_r;
      number c = std::sqrt( (gas_gamma-1.0) * (h - 0.5*v2) );
      number drho = W_r[density_component] - W_l[density_component];
      number dp = p_r - p_l;
      number dvn = v_r_normal - v_l_normal;
      
      number a1 = (dp - density * c * dvn) / (2.0*c*c);
      number a2 = drho - dp / (c*c);
      number a3 = (dp + density * c * dvn) / (2.0*c*c);

      number l1 = std::fabs(vel_normal - c);
      number l2 = std::fabs(vel_normal);
      number l3 = std::fabs(vel_normal + c);

      // entropy fix
      number delta = 0.1 * c;
      if(l1 < delta) l1 = 0.5 * (l1*l1/delta + delta);
      if(l3 < delta) l3 = 0.5 * (l3*l3/delta + delta);
      
      number Dflux[n_components];
      Dflux[density_component] = l1 * a1 + l2 * a2 + l3 * a3;
      Dflux[energy_component] = l1 * a1 * (h - c * vel_normal)
                              + l2 * a2 * 0.5 * v2
                              + l2 * density * (v_dot_dv - vel_normal * dvn)
                              + l3 * a3 * (h + c * vel_normal);
      normal_flux[density_component] = 0.5 * (W_l[density_component] * v_l_normal +
                                              W_r[density_component] * v_r_normal
                                              - Dflux[density_component]);
      normal_flux[energy_component] = 0.5 * (W_l[density_component] * h_l * v_l_normal +
                                             W_r[density_component] * h_r * v_r_normal
                                              - Dflux[energy_component]);
      number p_avg = 0.5 * (p_l + p_r);
      for(unsigned int d=0; d<dim; ++d)
      {
         Dflux[d] = (velocity[d] - normal[d] * c) * l1 * a1
                  + velocity[d] * l2 * a2
                  + (dv[d] - normal[d] * dvn) * l2 * density
                  + (velocity[d] + normal[d] * c) * l3 * a3;
         normal_flux[d] = normal[d] * p_avg
                        + 0.5 * (W_l[d] * v_l_normal + W_r[d] * v_r_normal)
                        - 0.5 * Dflux[d];
      }
   }
   
   
   // --------------------------------------------------------------------------
   // HLLC flux
   // Code borrowed from SU2 v2.0.2
   // --------------------------------------------------------------------------
   template <typename InputVector>
   static
   void hllc_flux
   (
    const dealii::Point<dim>         &normal,
    const InputVector                &W_l,
    const InputVector                &W_r,
    typename InputVector::value_type (&normal_flux)[n_components]
    )
   {
      typedef typename InputVector::value_type number;
      
      number rho_l_sqrt = std::sqrt(W_l[density_component]);
      number rho_r_sqrt = std::sqrt(W_r[density_component]);
      number fact_l = rho_l_sqrt / (rho_l_sqrt + rho_r_sqrt);
      number fact_r = 1.0 - fact_l;
      
      number v_l[dim], v_r[dim], velocity[dim];
      number v2_l = 0, v2_r = 0;
      number v_l_normal = 0, v_r_normal = 0;
      number vel_normal = 0, v2 = 0;
      for(unsigned int d=0; d<dim; ++d)
      {
         v_l[d]      = W_l[d] / W_l[density_component];
         v_r[d]      = W_r[d] / W_r[density_component];
         v2_l       += v_l[d] * v_l[d];
         v2_r       += v_r[d] * v_r[d];
         v_l_normal += v_l[d] * normal[d];
         v_r_normal += v_r[d] * normal[d];
         
         velocity[d] = v_l[d] * fact_l + v_r[d] * fact_r;
         vel_normal += velocity[d] * normal[d];
         v2         += velocity[d] * velocity[d];
      }
      
      //pressure
      number p_l = (gas_gamma-1) * (W_l[energy_component] - 0.5 * W_l[density_component] * v2_l);
      number p_r = (gas_gamma-1) * (W_r[energy_component] - 0.5 * W_r[density_component] * v2_r);
      
      // enthalpy
      number h_l = (W_l[energy_component] + p_l) / W_l[density_component];
      number h_r = (W_r[energy_component] + p_r) / W_r[density_component];

      // sound speed
      number c_l = std::sqrt(gas_gamma * p_l / W_l[density_component]);
      number c_r = std::sqrt(gas_gamma * p_r / W_r[density_component]);
      
      // energy per unit mass
      number e_l = W_l[energy_component] / W_l[density_component];
      number e_r = W_r[energy_component] / W_r[density_component];
      
      // roe average
      number h = h_l * fact_l + h_r * fact_r;
      number c = std::sqrt( (gas_gamma-1.0) * (h - 0.5*v2) );
      
      // speed of sound at l and r
      number s_l = std::min(vel_normal-c, v_l_normal-c_l);
      number s_r = std::min(vel_normal+c, v_r_normal+c_r);

      // speed of contact
      number s_m = (p_l - p_r
                    - W_l[density_component] * v_l_normal * (s_l-v_l_normal)
                    + W_r[density_component] * v_r_normal * (s_r-v_r_normal))
      /(W_r[density_component]*(s_r-v_r_normal) - W_l[density_component]*(s_l-v_l_normal));
      
      // Pressure at right and left (Pressure_j=Pressure_i) side of contact surface
      number pStar = W_r[density_component] * (v_r_normal-s_r)*(v_r_normal-s_m) + p_r;

      if (s_m >= 0.0) {
         if (s_l > 0.0)
         {
            normal_flux[density_component] = W_l[density_component]*v_l_normal;
            for (unsigned int d = 0; d < dim; d++)
               normal_flux[d] = W_l[density_component]*v_l[d]*v_l_normal + p_l*normal[d];
            normal_flux[energy_component] = e_l*W_l[density_component]*v_l_normal + p_l*v_l_normal;
         }
         else
         {
            number invSLmSs = 1.0/(s_l-s_m);
            number sLmuL = s_l-v_l_normal;
            number rhoSL = W_l[density_component]*sLmuL*invSLmSs;
            number rhouSL[dim];
            for (unsigned int d = 0; d < dim; d++)
               rhouSL[d] = (W_l[density_component]*v_l[d]*sLmuL+(pStar-p_l)*normal[d])*invSLmSs;
            number eSL = (sLmuL*e_l*W_l[density_component]-p_l*v_l_normal+pStar*s_m)*invSLmSs;
            
            normal_flux[density_component] = rhoSL*s_m;
            for (unsigned int d = 0; d < dim; d++)
               normal_flux[d] = rhouSL[d]*s_m + pStar*normal[d];
            normal_flux[energy_component] = (eSL+pStar)*s_m;
         }
      }
      else
      {
         if (s_r >= 0.0)
         {
            number invSRmSs = 1.0/(s_r-s_m);
            number sRmuR = s_r-v_r_normal;
            number rhoSR = W_r[density_component]*sRmuR*invSRmSs;
            number rhouSR[dim];
            for (unsigned int d = 0; d < dim; d++)
               rhouSR[d] = (W_r[density_component]*v_r[d]*sRmuR+(pStar-p_r)*normal[d])*invSRmSs;
            number eSR = (sRmuR*e_r*W_r[density_component]-p_r*v_r_normal+pStar*s_m)*invSRmSs;
            
            normal_flux[density_component] = rhoSR*s_m;
            for (unsigned int d = 0; d < dim; d++)
               normal_flux[d] = rhouSR[d]*s_m + pStar*normal[d];
            normal_flux[energy_component] = (eSR+pStar)*s_m;
         }
         else
         {
            normal_flux[density_component] = W_r[density_component]*v_r_normal;
            for (unsigned int d = 0; d < dim; d++)
               normal_flux[d] = W_r[density_component]*v_r[d]*v_r_normal + p_r*normal[d];
            normal_flux[energy_component] = e_r*W_r[density_component]*v_r_normal + p_r*v_r_normal;
         }
      }
      
   }
   
   //---------------------------------------------------------------------------
   // EulerEquations::compute_forcing_vector
   //---------------------------------------------------------------------------
   template <typename InputVector, typename number>
   static
   void compute_forcing_vector (const InputVector &W,
                                const dealii::Vector<double> &ext_force,
                                number            (&forcing)[n_components])
   {
      forcing[density_component] = 0.0;
      forcing[energy_component] = 0.0;
      
      //for(int d=0; d<dim; ++d)
      for(int d=0; d<v_components; ++d)
      {
         forcing[d] = W[density_component] * ext_force[d];
         forcing[energy_component] += W[d] * ext_force[d];
      }
      
      //for(int d=dim; d<2*dim; ++d)
      for(int d=v_components; d<2*v_components; ++d)
      {
         forcing[d] = 0;
      }
   }
   
   
   
   //---------------------------------------------------------------------------
   // Dealing with boundary conditions
   //---------------------------------------------------------------------------

   enum BoundaryKind
   {
      inflow_boundary,
      outflow_boundary,
      no_penetration_boundary,
      pressure_boundary,
      farfield_boundary,
      periodic
   };

   template <typename DataVector>
   static
   void
   compute_Wminus (const BoundaryKind           &boundary_kind,
                   const dealii::Tensor<1,dim>  &normal_vector,
                   const DataVector             &Wplus,
                   const dealii::Vector<double> &boundary_values,
                   const DataVector             &Wminus)
   {
      switch (boundary_kind)
      {
	      case inflow_boundary:
	      {
		for (unsigned int c = 0; c < n_components; ++c)
		  Wminus[c] = boundary_values(c);
		break;
	      }
            
	      case outflow_boundary:
	      {
		for (unsigned int c = 0; c < n_components; ++c)
		  Wminus[c] = Wplus[c];
		break;
	      }
            
	      case pressure_boundary:
	      {
		const typename DataVector::value_type
		density = Wplus[density_component];
		
		typename DataVector::value_type kinetic_energy = 0;
		//for (unsigned int d=0; d<dim; ++d)
		for (unsigned int d=0; d<v_components; ++d)
		  kinetic_energy += Wplus[d]*Wplus[d];
		kinetic_energy *= 0.5/density;
		
		// Same as in the Euler equations, making the magnetic field also equal to the interior values
		//for (unsigned int c = 0; c < 2*dim; ++c)
		for (unsigned int c = 0; c < v_components; ++c)
		  Wminus[magnetic_component+c] = Wplus[magnetic_component+c];
		
		Wminus[density_component] = density;
		Wminus[energy_component] = boundary_values(energy_component) / (gas_gamma-1.0) +
                        kinetic_energy;
			
		break;
	      }
            
	      case no_penetration_boundary:
	      {
		typename DataVector::value_type
		vdotn = 0;
		for (unsigned int d = 0; d < dim; d++)
		  vdotn += Wplus[d]*normal_vector[d];
		for (unsigned int c = 0; c < dim; ++c)
		  Wminus[c] = Wplus[c] - 2.0 * vdotn * normal_vector[c];
		
		// Magnetic field doesn't change here either
		for (unsigned int c = 0; c < v_components; ++c)
		  Wminus[magnetic_component+c] = Wplus[magnetic_component+c];
		
		Wminus[density_component] = Wplus[density_component];
		Wminus[energy_component]  = Wplus[energy_component];
		break;
		
	      }
            
	      case farfield_boundary:
	      {
		for (unsigned int c = 0; c < n_components; ++c)
		  Wminus[c] = boundary_values(c);
		break;
	      }
	      
	      default:
		Assert (false, dealii::ExcNotImplemented());
      }

   }
   
   //TODO: Check if the entropy_var is the same for MHD
   //---------------------------------------------------------------------------
   // Compute entropy variables V, given conserved variables W
   //---------------------------------------------------------------------------
   template <typename InputVector, typename number>
   static
   void entropy_var (const InputVector &W,
                     number            (&V)[n_components])
   {
      number pressure = compute_pressure<number> (W);
      number T = pressure / W[density_component];

      number u2 = 0;
      //for(unsigned int d=0; d<dim; ++d)
      for(unsigned int d=0; d<v_components; ++d)
      {
         number u = W[d] / W[density_component];
         V[d] = u / T;
         u2 += u * u;
      }

      V[density_component] = log(W[density_component] / std::pow(T, 1.0/(gas_gamma-1.0))) 
                           - 0.5 * u2 / T;
      V[energy_component] = -1.0 / T;
   }
   
   class Postprocessor : public dealii::DataPostprocessor<dim>
   {
   public:
      Postprocessor (const bool do_schlieren_plot);
      
      virtual
      void
      compute_derived_quantities_vector 
         (const std::vector<dealii::Vector<double> >              &uh,
          const std::vector<std::vector<dealii::Tensor<1,dim> > > &duh,
          const std::vector<std::vector<dealii::Tensor<2,dim> > > &dduh,
          const std::vector<dealii::Point<dim> >                  &normals,
          const std::vector<dealii::Point<dim> >                  &evaluation_points,
          std::vector<dealii::Vector<double> >                    &computed_quantities) const;
      
      virtual std::vector<std::string> get_names () const;
      
      virtual
      std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
      get_data_component_interpretation () const;
      
      virtual dealii::UpdateFlags get_needed_update_flags () const;
      
      virtual unsigned int n_output_variables() const;
      
   private:
      const bool do_schlieren_plot;
   };
};

#endif
