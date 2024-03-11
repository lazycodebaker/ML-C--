 
#include "include_file.h"

// Min-Max Normalisation z = (x - min) / (max - min)
template <typename T>
std::vector<double> min_max_normalisation(const std::vector<T> &data)
{
    // double min = *std::min_element(data.begin(), data.end());
    // double max = *std::max_element(data.begin(), data.end());
    const auto [min, max] = std::minmax_element(data.begin(), data.end());

    std::vector<double> normalized_data;
    normalized_data.reserve(data.size());

    const double range = *max - *min;
    for (const T &value : data)
    {
        normalized_data.push_back((static_cast<double>(value) - *min) / range);
    };
    return normalized_data;
}

// Standardisation (Standard Scaler) z = (x - mean) / stdev
template <typename T>
std::vector<double> standardisation(const std::vector<T> &data)
{
    const double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    const double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
    const double stdev = std::sqrt(sq_sum / data.size() - mean * mean);

    std::vector<double> standardized_data; // --> container z-score or standardised values
    standardized_data.reserve(data.size());

    /*
        std::vector<double>::iterator it;
        for (it = data.begin(); it != data.end(); it++) {
            *it = (*it - mean) / stdev;
        };
    */

    std::transform(data.begin(), data.end(), std::back_inserter(standardized_data),
                   [mean, stdev](const auto &value)
                   { return (value - mean) / stdev; });

    return standardized_data;
}

// MAX-ABS Normalisation z = x / max(abs(x))
// used in sparse data and image processing , where data is already centered at zero ( where there is more zero values )
template <typename T>
std::vector<double> max_abs_normalisation(const std::vector<T> &data)
{
    const auto max_abs = std::max_element(data.begin(), data.end(), [](const auto &lhs, const auto &rhs)
                                          { return std::abs(lhs) < std::abs(rhs); });

    std::vector<double> normalized_data;
    normalized_data.reserve(data.size());

    for (const T &value : data)
    {
        normalized_data.push_back(static_cast<double>(value) / *max_abs);
    };
    return normalized_data;
}

// MEAN Normalisation z = (x - mean) / (max - min)
template <typename T>
std::vector<double> mean_normalisation(const std::vector<T> &data)
{
    const double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    const auto [min, max] = std::minmax_element(data.begin(), data.end());

    std::vector<double> normalized_data;
    normalized_data.reserve(data.size());

    const double range = *max - *min;
    for (const T &value : data)
    {
        normalized_data.push_back((static_cast<double>(value) - mean) / range);
    };
    return normalized_data;
}


template <typename T>
void print_vector(const std::vector<T> &data)
{
    /*
        std::vector<double>::iterator it;
        for (it = data.begin(); it != data.end(); it++) {
            std::cout << *it << std::endl;
        };
    */
    for (const auto &value : data)
    {
        std::cout << value << std::endl;
    }
}

int main()
{
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Min-Max Normalisation
    std::cout << "Min-Max Normalisation:" << std::endl;
    print_vector(min_max_normalisation(data));

    // Standardisation
    std::cout << "\nStandardisation:" << std::endl;
    print_vector(standardisation(data));

    return EXIT_SUCCESS;
}
