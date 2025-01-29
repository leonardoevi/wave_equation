#include "WaveSolver.hpp"

void WaveSolver::setup() {
  // Create the mesh.
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
    rhs_matrix.reinit(sparsity);

    // Initializing the system right-hand side
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    // Initializing the solution vector
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution_owned_old.reinit(locally_owned_dofs, MPI_COMM_WORLD);
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
      const double mu_loc = mu.value(fe_values.quadrature_point(q));

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          cell_mass_matrix(i, j) += fe_values.shape_value(i, q) *
                                    fe_values.shape_value(j, q) /
                                    (deltat * deltat) * fe_values.JxW(q);

          cell_stiffness_matrix(i, j) += mu_loc *
                                         fe_values.shape_grad(i, q) *
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

  // We build the matrix on the left-hand side of the algebraic problem (the one that we'll invert at each timestep).
  lhs_matrix.copy_from(mass_matrix);
  lhs_matrix.add(1, stiffness_matrix);

  // We build the matrix on the right-hand side (the one that multiplies the old solution un).
  rhs_matrix.copy_from(mass_matrix);
}

void WaveSolver::assemble_rhs(const double &time) {
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe, *quadrature, update_values | update_quadrature_points | update_JxW_values);

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_rhs = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);

    cell_rhs = 0.0;

    for (unsigned int q = 0; q < n_q; ++q) {
      // Compute f(tn+1)
      forcing_term.set_time(time);
      const double f_new_loc = forcing_term.value(fe_values.quadrature_point(q));

      // Compute f(tn)
      forcing_term.set_time(time - deltat);
      const double f_old_loc = forcing_term.value(fe_values.quadrature_point(q));

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        // remove theta from here
        int theta = 1;
        cell_rhs(i) += (theta * f_new_loc + (1.0 - theta) * f_old_loc) * fe_values.shape_value(i, q) * fe_values.JxW(q);
      }
    }

    cell->get_dof_indices(dof_indices);
    system_rhs.add(dof_indices, cell_rhs);
  }

  system_rhs.compress(VectorOperation::add);

  // Add the term that comes from the old solution. (F + M/deltat^2 * (2 * u_k-1 - u_k-2) )
  // 2 * solution_owned - solution_owned_old
  solution_owned.sadd(2.0, -1.0, solution_owned_old);

  rhs_matrix.vmult_add(system_rhs, solution_owned);

  // Boundary conditions.
  {
    // We construct a map that stores, for each DoF corresponding to a
    // Dirichlet condition, the corresponding value. E.g., if the Dirichlet
    // condition is u_i = b, the map will contain the pair (i, b).
    std::map<types::global_dof_index, double> boundary_values;

    // Then, we build a map that, for each boundary tag, stores the
    // corresponding boundary function.
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    Functions::ConstantFunction<dim> function_zero(0.0);

    boundary_functions[0] = &function_zero;
    boundary_functions[1] = &function_zero;
    boundary_functions[2] = &function_zero;
    boundary_functions[3] = &function_zero;

    // interpolate_boundary_values fills the boundary_values map.
    VectorTools::interpolate_boundary_values(dof_handler, boundary_functions, boundary_values);

    // Finally, we modify the linear system to apply the boundary
    // conditions. This replaces the equations for the boundary DoFs with
    // the corresponding u_i = 0 equations.
    MatrixTools::apply_boundary_values(boundary_values, lhs_matrix, solution, system_rhs, true);
  }
}

void WaveSolver::solve_time_step() {
  SolverControl solver_control(1000, 1e-12 * system_rhs.l2_norm());

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionSSOR      preconditioner;
  preconditioner.initialize(lhs_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  solution_owned_old = solution;

  solver.solve(lhs_matrix, solution_owned, system_rhs, preconditioner);
  pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;

  solution = solution_owned;
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

    VectorTools::interpolate(dof_handler, u_0, solution_owned_old);
    solution = solution_owned_old;

    // Output the initial solution.
    output(0);

    VectorTools::interpolate(dof_handler, u_1, solution_owned);
    // solution += deltat * u_1 + u_0
    solution_owned.sadd(deltat, solution_owned);
    solution_owned.add(1, solution_owned_old);

    solution = solution_owned;
    output(1);

    pcout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0 + 1;
  double       time      = 0 + deltat;

  while (time < T) {
    time += deltat;
    ++time_step;

    pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5) << time << ":" << std::flush;

    assemble_rhs(time);
    solve_time_step();
    output(time_step);
  }
}