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
void ConservationLaw<dim>::alfven_wave_grid ()
{
  pcout<<"\n\t Creating grid for the Alfven wave test case\n";
  
  double ly = std::sqrt(5), lx = 0.5*ly;

  Triangulation<dim> tria;
  
  const Point<dim> p1 (0,0);
  const Point<dim> p2 (lx, ly);
  std::vector<unsigned int> vect (dim);
  vect[0]=3;
  vect[1]=2*vect[0];
  
  const std::vector<unsigned int> repetition = vect;

  GridGenerator::subdivided_hyper_rectangle (tria,
					     repetition,
					     p1,
					     p2,
					     true);
//*--------------------check---------------*/
  Triangulation<2>::active_cell_iterator
  cell = triangulation.begin_active(),
  endc = triangulation.end();
  for (; cell!=endc; ++cell)
  {
    if(cell->at_boundary())
    {
      for(unsigned int i=0; i<)
    }
  }
//*--------------------check---------------*/
  
  std::ofstream out ("grid.msh");
  GridOut grid_out;
  GridOutFlags::Msh flags(true,true);
  grid_out.set_flags(flags);
  grid_out.write_msh (tria, out);
  pcout << "Grid written to grid.msh" << std::endl;
  
  std::ofstream out1 ("grid.eps");
  GridOut grid_out1;
  grid_out1.write_eps (tria, out1);
  pcout << "Grid written to grid.eps" << std::endl;
  
  
  triangulation.copy_triangulation(tria);
  
}

template class ConservationLaw<2>;