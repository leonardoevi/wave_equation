#include "WaveSolver.hpp"

int main(int argc, char *argv[]) {
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    const std::string  mesh_file_name = "../mesh/mesh-square-h0.012500.msh";

    constexpr unsigned int degree = 1;
    constexpr double T            = 10;
    constexpr double deltat       = 0.01;

    WaveSolver problem(mesh_file_name, degree, T, deltat);
    problem.setup();
    problem.solve();

    return 0;
}