#include "WaveSolver.hpp"
#include <deal.II/fe/mapping_fe.h>

void WaveSolver::setup() {
  // Create the mesh.
#if DIM == 1
  {
    std::cout << "Initializing the mesh" << std::endl;
    GridGenerator::subdivided_hyper_cube(mesh, 50 , 0.0, 2 * M_PI, true);
    std::cout << "  Number of elements = " << mesh.n_active_cells()
              << std::endl;

    // Write the mesh to file.
    const std::string mesh_file_name = "mesh-" + std::to_string(500) + ".vtk";
    GridOut           grid_out;
    std::ofstream     grid_out_file(mesh_file_name);
    grid_out.write_vtk(mesh, grid_out_file);
    std::cout << "  Mesh saved to " << mesh_file_name << std::endl;
  }
#else
{
  pcout << "Initializing the mesh" << std::endl;

  Triangulation<dim> mesh_serial;

  GridIn<dim> grid_in;
  grid_in.attach_triangulation(mesh_serial);

  std::ifstream grid_in_file(mesh_file_name);
  grid_in.read_msh(grid_in_file);

  GridTools::partition_triangulation(mpi_size, mesh_serial);
  const auto construction_data = TriangulationDescription::Utilities::
    create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
  mesh.create_triangulation(construction_data);

  pcout << "  Number of elements = " << mesh.n_global_active_cells()
        << std::endl;
}
#endif

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  // Initialize the linear system.
  {
    // Initializing the sparsity pattern
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    // Initializing the matrices
    mass_matrix.reinit(sparsity);
    stiffness_matrix.reinit(sparsity);
    lhs_matrix.reinit(sparsity);

    // Initializing the system right-hand side
    f_k.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    f_k_old.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    // Initializing the solution vector
    u_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    u_old_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    v_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    v_old_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    tmp.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    b.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  }
}

void WaveSolver::assemble_matrices() {
  pcout << "===============================================" << std::endl;
  pcout << "Assembling the system matrices" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe, *quadrature, update_values | update_gradients |update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  mass_matrix      = 0.0;
  stiffness_matrix = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);

    cell_mass_matrix      = 0.0;
    cell_stiffness_matrix = 0.0;

    for (unsigned int q = 0; q < n_q; ++q) {
      // Evaluate coefficients on this quadrature node.

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          cell_mass_matrix(i, j) += fe_values.shape_value(i, q) *
                                    fe_values.shape_value(j, q) * fe_values.JxW(q);

          cell_stiffness_matrix(i, j) += fe_values.shape_grad(i, q) *
                                         fe_values.shape_grad(j, q) * fe_values.JxW(q);
        }
      }
    }

    cell->get_dof_indices(dof_indices);

    mass_matrix.add(dof_indices, cell_mass_matrix);
    stiffness_matrix.add(dof_indices, cell_stiffness_matrix);
  }

  mass_matrix.compress(VectorOperation::add);
  stiffness_matrix.compress(VectorOperation::add);
}

void WaveSolver::compute_F(const double &time) {
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe, *quadrature, update_values | update_quadrature_points | update_JxW_values);

  Vector<double> cell_f_k(dofs_per_cell);
  Vector<double> cell_f_k_old(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  f_k = 0.0;
  f_k_old = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);

    cell_f_k = 0.0;
    cell_f_k_old = 0.0;

    for (unsigned int q = 0; q < n_q; ++q) {
      // Compute f(tn+1)
      forcing_term.set_time(time);
      const double f_loc = forcing_term.value(fe_values.quadrature_point(q));

      // Compute f(tn)
      forcing_term.set_time(time - deltat);
      const double f_loc_old = forcing_term.value(fe_values.quadrature_point(q));

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        cell_f_k_old(i) += f_loc_old * fe_values.shape_value(i, q) * fe_values.JxW(q);
        cell_f_k(i) += f_loc * fe_values.shape_value(i, q) * fe_values.JxW(q);
      }
    }

    cell->get_dof_indices(dof_indices);
    f_k.add(dof_indices, cell_f_k);
    f_k_old.add(dof_indices, cell_f_k_old);
  }

  f_k.compress(VectorOperation::add);
  f_k_old.compress(VectorOperation::add);
}

void WaveSolver::solve_time_step_BE(const double &time) {
  // compute system rhs for u_n+1 computation
  {
    b = 0.0;

    // b = M * u_n
    mass_matrix.vmult_add(b, u_old_owned);

    // tmp = delta_t^2 * F_k+1
    tmp = f_k;
    tmp *= (deltat * deltat);

    b.add(tmp);

    // tmp = delta_t * M * v_n
    mass_matrix.vmult(tmp, v_old_owned);
    b.add(deltat, tmp);
  }

  lhs_matrix.copy_from(mass_matrix);
  lhs_matrix.add(deltat*deltat, stiffness_matrix);
  // Boundary conditions.
  {
    // We construct a map that stores, for each DoF corresponding to a
    // Dirichlet condition, the corresponding value. E.g., if the Dirichlet
    // condition is u_i = b, the map will contain the pair (i, b).
    std::map<types::global_dof_index, double> boundary_values;

    // Then, we build a map that, for each boundary tag, stores the
    // corresponding boundary function.
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    //Functions::ZeroFunction<dim> function_zero;
    FunctionU<dim> e{};
    e.set_time(time);

    // TODO change it when changing dimension/mesh. Now it works for 2D square centered mesh
    boundary_functions[0] = &e;
    boundary_functions[1] = &e;
    boundary_functions[2] = &e;
    boundary_functions[3] = &e;

    // interpolate_boundary_values fills the boundary_values map.
    VectorTools::interpolate_boundary_values(dof_handler, boundary_functions, boundary_values);

    // Finally, we modify the linear system to apply the boundary
    // conditions. This replaces the equations for the boundary DoFs with
    // the corresponding u_i = 0 equations.
    MatrixTools::apply_boundary_values(boundary_values, lhs_matrix, u_owned, b, true);
  }

  // solve for u_n+1 (u_owned)
  {
    SolverControl solver_control(1000, 1e-8 * b.l2_norm());

    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
    TrilinosWrappers::PreconditionSSOR      preconditioner;
    preconditioner.initialize(lhs_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

    solver.solve(lhs_matrix, u_owned, b, preconditioner);
    //pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;
  }

  // compute system rhs for v_n+1 computation
  {
    b = 0.0;

    // b = M * v_k
    mass_matrix.vmult(b, v_old_owned);

    // tmp = delta_t * F_k
    tmp = f_k;
    tmp *= deltat;
    b.add(tmp);

    // tmp = A * u_k+1
    stiffness_matrix.vmult(tmp, u_owned);
    tmp *= (-deltat);
    b.add(tmp);
  }

  // boundary condition on vn+1
  lhs_matrix.copy_from(mass_matrix);
  /*{
    // We construct a map that stores, for each DoF corresponding to a
    // Dirichlet condition, the corresponding value. E.g., if the Dirichlet
    // condition is u_i = b, the map will contain the pair (i, b).
    std::map<types::global_dof_index, double> boundary_values;

    // Then, we build a map that, for each boundary tag, stores the
    // corresponding boundary function.
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    //Functions::ConstantFunction<dim> function_zero(1.0);
    FunctiondU<dim> e{};
    e.set_time(time);

    //boundary_functions[0] = &e;
    //boundary_functions[1] = &e;
    //boundary_functions[2] = &e;
    //boundary_functions[3] = &e;

    // interpolate_boundary_values fills the boundary_values map.
    VectorTools::interpolate_boundary_values(dof_handler, boundary_functions, boundary_values);

    // Finally, we modify the linear system to apply the boundary
    // conditions. This replaces the equations for the boundary DoFs with
    // the corresponding u_i = 0 equations.
    MatrixTools::apply_boundary_values(boundary_values, lhs_matrix, v_owned, b, true);
  }*/

  // solve for v_n+1 (v_owned)
  {
    SolverControl solver_control(1000, 1e-8 * b.l2_norm());

    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
    TrilinosWrappers::PreconditionSSOR      preconditioner;
    preconditioner.initialize(lhs_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

    solver.solve(lhs_matrix, v_owned, b, preconditioner);
    // pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;
  }

  solution = u_owned;
}

void WaveSolver::solve_time_step_LF(const double &time) {
  // compute system rhs for u_n+1 computation
  {
    b = 0.0;

    // b = M * u_n
    mass_matrix.vmult_add(b, u_old_owned);

    // tmp = A * u_n
    stiffness_matrix.vmult(tmp, u_old_owned);
    tmp *= (-deltat * deltat / 2.0);

    b.add(tmp);

    // tmp = M * v_n
    mass_matrix.vmult(tmp, v_old_owned);
    b.add(deltat, tmp);

    b.add(deltat*deltat/2.0, f_k_old);
  }

  // Boundary conditions.
  lhs_matrix.copy_from(mass_matrix);
  {
    // We construct a map that stores, for each DoF corresponding to a
    // Dirichlet condition, the corresponding value. E.g., if the Dirichlet
    // condition is u_i = b, the map will contain the pair (i, b).
    std::map<types::global_dof_index, double> boundary_values;

    // Then, we build a map that, for each boundary tag, stores the
    // corresponding boundary function.
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    Functions::ZeroFunction<dim> function_zero;
    FunctionU<dim> e{};
    e.set_time(time);

    // TODO change it when changing dimension/mesh. Now it works for 2D square centered mesh
    boundary_functions[0] = &e;
    boundary_functions[1] = &function_zero;
    //boundary_functions[2] = &e;
    //boundary_functions[3] = &e;

    // interpolate_boundary_values fills the boundary_values map.
    VectorTools::interpolate_boundary_values(dof_handler, boundary_functions, boundary_values);

    // Finally, we modify the linear system to apply the boundary
    // conditions. This replaces the equations for the boundary DoFs with
    // the corresponding u_i = 0 equations.
    MatrixTools::apply_boundary_values(boundary_values, lhs_matrix, u_owned, b, true);
  }

  // solve for u_n+1 (u_owned)
  {
    SolverControl solver_control(1000, 1e-8 * b.l2_norm());

    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
    TrilinosWrappers::PreconditionSSOR      preconditioner;
    preconditioner.initialize(lhs_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

    solver.solve(lhs_matrix, u_owned, b, preconditioner);
    //pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;
  }

  // compute system rhs for v_n+1 computation
  {
    b = 0.0;

    mass_matrix.vmult(b, v_old_owned);

    tmp = u_old_owned;
    tmp.add(u_owned);
    tmp *= (-deltat/2.0);
    stiffness_matrix.vmult_add(b, tmp);

    tmp = f_k_old;
    tmp.add(f_k);
    tmp *= (deltat/2.0);
    b.add(tmp);
  }

  // boundary condition on vn+1
  lhs_matrix.copy_from(mass_matrix);
  /*{
    // We construct a map that stores, for each DoF corresponding to a
    // Dirichlet condition, the corresponding value. E.g., if the Dirichlet
    // condition is u_i = b, the map will contain the pair (i, b).
    std::map<types::global_dof_index, double> boundary_values;

    // Then, we build a map that, for each boundary tag, stores the
    // corresponding boundary function.
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    //Functions::ConstantFunction<dim> function_zero(1.0);
    FunctiondU<dim> e{};
    e.set_time(time);

    boundary_functions[0] = &e;
    boundary_functions[1] = &e;
    boundary_functions[2] = &e;
    boundary_functions[3] = &e;

    // interpolate_boundary_values fills the boundary_values map.
    VectorTools::interpolate_boundary_values(dof_handler, boundary_functions, boundary_values);

    // Finally, we modify the linear system to apply the boundary
    // conditions. This replaces the equations for the boundary DoFs with
    // the corresponding u_i = 0 equations.
    MatrixTools::apply_boundary_values(boundary_values, lhs_matrix, v_owned, b, true);
  }*/

  // solve for v_n+1 (v_owned)
  {
    SolverControl solver_control(1000, 1e-8 * b.l2_norm());

    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
    TrilinosWrappers::PreconditionSSOR      preconditioner;
    preconditioner.initialize(lhs_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

    solver.solve(lhs_matrix, v_owned, b, preconditioner);
    //pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;
  }

  solution = u_owned;
}

void WaveSolver::output(const unsigned int &time_step) const {

  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "u");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record(
    "./", "output", time_step, MPI_COMM_WORLD, 3);
}

void WaveSolver::solve() {
  assemble_matrices();

  pcout << "===============================================" << std::endl;

  // Apply the initial condition.
  {
    pcout << "Applying the initial condition" << std::endl;

    VectorTools::interpolate(dof_handler, u_0, u_old_owned);
    VectorTools::interpolate(dof_handler, v_0, v_old_owned);

    solution = u_old_owned;
    output(0);

    pcout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;
  double       time      = 0;

  while (time < T) {
    time += deltat;
    ++time_step;

    if (time_step % static_cast<int>((T / deltat) * 0.1) == 0) pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5) << time << std::endl;

    compute_F(time);
    #if LEAP_FROG
    solve_time_step_LF(time);
    #else
    solve_time_step_BE(time);
    #endif

    if (time_step % SKIPS == 0) output(time_step);

    u_old_owned = u_owned;
    v_old_owned = v_owned;
  }

  endT = time;
}

double WaveSolver::compute_error(const VectorTools::NormType &norm_type) const {
  FE_SimplexP<dim> fe_linear(1);
  MappingFE        mapping(fe_linear);

  // The error is an integral, and we approximate that integral using a
  // quadrature formula. To make sure we are accurate enough, we use a
  // quadrature formula with one node more than what we used in assembly.
  const QGaussSimplex<dim> quadrature_error(r + 2);

  FunctionU<dim> exact_sol{};
  exact_sol.set_time(endT);

  // First we compute the norm on each element, and store it in a vector.
  Vector<double> error_per_cell(mesh.n_active_cells());
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    exact_sol,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  // Then, we add out all the cells.
  const double error =
    VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}