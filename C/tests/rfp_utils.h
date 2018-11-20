#include <blas_configuration.h>
#include <utility>

// Utility file for tests in creating matrices.

/*! Creates a pair of symmetric or triangular matrices of the given size, the first in full format and the
 *	second in RFP format.
 *
 * @param n: The size of the matrix to create.
 *
 */
std::pair<unique_aligned_array<double>, unique_aligned_array<double>> make_random_matrices(int n, bool triangular = false);
