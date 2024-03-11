// Cauchy-Schwarz Inequality
/*
Cauchy-Schwarz Inequality :

The Cauchy-Schwarz inequality states that the sum of the squares of the
absolute values of the eigenvalues of a matrix is less than or equal to
the product of the eigenvalues.
*/

#include "iostream"
#include "algorithm"
#include "vector"
#include "numeric"
#include "cmath"
#include "unordered_map"
#include "vector"
#include "algorithm"
#include "execution"
#include "utility"
#include "random"
#include "limits"
#include "stdexcept"

std::vector<int> U = std::vector<int>{1, -2, 4};
std::vector<int> V = std::vector<int>{2, 4, -2};

void cauchySchwarzInequality()
{
    std::cout << "Cauchy-Schwarz Inequality : " << std::endl;

    // U.V
    const int lhs_result = std::abs(std::inner_product(U.begin(), U.end(), V.begin(), 0));

    // sqrt(U^2) . sqrt(V^2)
    const int result1 = std::inner_product(U.begin(), U.end(), U.begin(), 0);
    const int result2 = std::inner_product(V.begin(), V.end(), V.begin(), 0);

    const double rhs_result = std::abs(std::sqrt(result1) * std::sqrt(result2));
    std::cout << "LHS = | U . V | = " << lhs_result << std::endl;
    std::cout << "RHS = | sqrt(U^2) . sqrt(V^2) | = " << (rhs_result) << std::endl;
}

int main()
{
    cauchySchwarzInequality();
    return EXIT_SUCCESS;
};