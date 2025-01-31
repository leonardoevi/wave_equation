#ifndef WAVE_SOLVER_HPP
#define WAVE_SOLVER_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include "Functions.hpp"

#include <fstream>
#include <iostream>
#include<cmath>


using namespace dealii;

class WaveSolver
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = DIM;

  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  WaveSolver(const std::string  &mesh_file_name_,
       const unsigned int &r_,
       const double       &T_,
       const double       &deltat_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , T(T_)
    , mesh_file_name(mesh_file_name_)
    , r(r_)
    , deltat(deltat_)
    , mesh(MPI_COMM_WORLD)
  {}

  // Initialization.
  void
  setup();

  // Solve the problem.
  void
  solve();

  double
  compute_error(const VectorTools::NormType &norm_type) const;

protected:
  // Assemble the mass and stiffness matrices.
  void
  assemble_matrices();

  // Assemble the right-hand side of the problem.
  void
  assemble_rhs(const double &time);

  // Output.
  void
  output(const unsigned int &time_step) const;

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Problem definition. ///////////////////////////////////////////////////////

  // Forcing term.
  ForcingTerm<dim> forcing_term;

  // Initial condition.
  FunctionU0<dim> u_0;
  FunctionV0<dim> v_0;

  // Final time.
  const double T;

  // Discretization. ///////////////////////////////////////////////////////////

  // Mesh file name.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Time step.
  const double deltat;

  // Final simul time
  double endT = 0;

  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // Mass matrix M / deltat.
  TrilinosWrappers::SparseMatrix mass_matrix;

  // Stiffness matrix A.
  TrilinosWrappers::SparseMatrix stiffness_matrix;

  TrilinosWrappers::SparseMatrix lhs_matrix;

  // Right-hand side vector in the linear system.
  TrilinosWrappers::MPI::Vector f_k;
  TrilinosWrappers::MPI::Vector f_k_old;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector u_owned;
  TrilinosWrappers::MPI::Vector u_old_owned;

  TrilinosWrappers::MPI::Vector v_owned;
  TrilinosWrappers::MPI::Vector v_old_owned;

  TrilinosWrappers::MPI::Vector tmp;
  TrilinosWrappers::MPI::Vector b;
};

#endif