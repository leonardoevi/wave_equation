#include "WaveSolver.hpp"
#include <deal.II/base/convergence_table.h>

#define CONVERGENCE false

int main(int argc, char *argv[]) {
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

#if CONVERGENCE
    ConvergenceTable table;

    const std::vector<std::string>  mesh_file_name = {
        "../mesh/squarec0_8000.msh",
        "../mesh/squarec0_4000.msh",
        "../mesh/squarec0_2000.msh",
        "../mesh/squarec0_1000.msh",
        "../mesh/squarec0_0500.msh"};

    const std::vector<double>      h_vals = {
        0.8,
        0.4,
        1.0 / 5.0,
        1.0 / 10.0,
        1.0 / 20.0};

    constexpr unsigned int degree = 1;
    constexpr double T            = 10;
    constexpr double deltat       = 0.1;

    const std::vector<double>      delta_ts = {
        deltat / 1.0,
        deltat / 2.0,
        deltat / 4.0,
        deltat / 8.0,
        deltat / 16.0};

    std::ofstream convergence_file("convergence.csv");
    convergence_file << "h,eL2" << std::endl;

    for (unsigned int i = 0; i < mesh_file_name.size(); i++) {
        WaveSolver problem(mesh_file_name[i], degree, T, delta_ts[i]);
        problem.setup();
        problem.solve();

        const double error_L2 = problem.compute_error(VectorTools::L2_norm);

        table.add_value("h", h_vals[i]);
        table.add_value("L2", error_L2);

        convergence_file << h_vals[i] << "," << error_L2
                         << std::endl;
    }

    table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
    table.set_scientific("L2", true);
    table.write_text(std::cout);
#else
    const std::string  mesh_file_name = "../mesh/squarec0_0500.msh";

    constexpr unsigned int degree = 1;
    constexpr double T            = 10;
    constexpr double deltat       = 0.1;

    WaveSolver problem(mesh_file_name, degree, T, deltat);
    problem.setup();
    problem.solve();
#endif

    return 0;
}