

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

// finding the angle between vectors
// a triangle with sides A , B , and angle between then theta , the opposite of thete
// is a - b vector == c
// c^2 = a^2 + b^2 - 2 * a * b * cos(theta)
// theta = acos( (a^2 + b^2 - c^2) / (2 * a * b) )

double findVectorAngle(const std::vector<int> &a, const std::vector<int> &b)
{
    if (a.size() != b.size())
    {
        throw std::invalid_argument("Vectors must be of equal size");
    }

    double dotProduct = std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
    double normA = std::inner_product<std::vector<int>::const_iterator, std::vector<int>::const_iterator, double>(a.begin(), a.end(), a.begin(), 0.0);
    double normB = std::inner_product(b.begin(), b.end(), b.begin(), 0.0);

    normA = std::sqrt(normA);
    normB = std::sqrt(normB);

    if (normA == 0 || normB == 0)
    {
        throw std::invalid_argument("Vectors cannot have zero magnitude");
    }

    double cosTheta = dotProduct / (normA * normB);
    return std::acos(cosTheta);
}

int main()
{
    try
    {
        std::vector<int> a = {1, 2, 3};
        std::vector<int> b = {4, 5, 6};

        double angle = findVectorAngle(a, b);
        std::cout << "Angle between vectors a and b: " << angle << " radians\n";
        std::cout << "Angle between vectors a and b: " << angle * 180.0 / M_PI << " degrees\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}