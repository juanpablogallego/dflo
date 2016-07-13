#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

#include "claw.h"

using namespace dealii;

template <int dim>
void ConservationLaw<dim>::alfven_wave_grid () //parallel::distributed::Triangulation<dim> &triangulation)
{
  double ly = std::sqrt(5), lx = 0.5*ly;

  Triangulation<dim> tria;
  
  const Point<dim> p1 (0,0);
  const Point<dim> p2 (lx, ly);
  std::vector<unsigned int> vect (dim);
  vect[0]=128;
  vect[1]=2*vect[0];
  
  const std::vector<unsigned int> repetition = vect;

  GridGenerator::subdivided_hyper_rectangle (tria,
					     repetition,
					     p1,
					     p2,
					     true);
  //tria.refine_global (4);
  
  triangulation.copy_triangulation(tria);
  
}

template class ConservationLaw<2>;