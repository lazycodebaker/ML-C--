

// LOG Transform , reciprocal transform , square root transform ML Functions

#include "include_file.h"

class MLTransformer
{
public:
    std::vector<double> standardize(const std::vector<double> &data)
    {
        double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
        double sumSquaredDiff = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
        double stddev = std::sqrt(sumSquaredDiff / data.size() - mean * mean);

        std::vector<double> transformedData(data.size());
        std::transform(data.begin(), data.end(), transformedData.begin(),
                       [&](double val)
                       { return (val - mean) / stddev; });

        return transformedData;
    }

    std::vector<double> minMaxScale(const std::vector<double> &data, double minVal, double maxVal)
    {
        double minData = *std::min_element(data.begin(), data.end());
        double maxData = *std::max_element(data.begin(), data.end());

        std::vector<double> transformedData(data.size());
        std::transform(data.begin(), data.end(), transformedData.begin(),
                       [&](double val)
                       { return (val - minData) / (maxData - minData) * (maxVal - minVal) + minVal; });

        return transformedData;
    }

    std::vector<double> featureHashing(const std::vector<std::string> &data, size_t numFeatures)
    {
        std::vector<double> hashedData(numFeatures, 0.0);
        for (const auto &item : data)
        {
            size_t hash = std::hash<std::string>{}(item) % numFeatures;
            hashedData[hash]++;
        }
        return hashedData;
    }

    std::vector<std::vector<double>> addPolynomialFeatures(const std::vector<double> &data, size_t degree)
    {
        std::vector<std::vector<double>> polynomialFeatures;
        for (const auto &val : data)
        {
            std::vector<double> features;
            for (size_t d = 1; d <= degree; ++d)
            {
                features.push_back(std::pow(val, d));
            }
            polynomialFeatures.push_back(features);
        }
        return polynomialFeatures;
    }

    std::vector<std::vector<double>> oneHotEncode(const std::vector<std::string> &categories)
    {
        std::unordered_map<std::string, size_t> categoryIndices;
        size_t index = 0;
        for (const auto &category : categories)
        {
            if (categoryIndices.find(category) == categoryIndices.end())
            {
                categoryIndices[category] = index++;
            }
        }

        std::vector<std::vector<double>> encodedCategories(categories.size(), std::vector<double>(index, 0.0));
        for (size_t i = 0; i < categories.size(); ++i)
        {
            encodedCategories[i][categoryIndices[categories[i]]] = 1.0;
        }

        return encodedCategories;
    };

    std::vector<double> logTransform(const std::vector<double> &data)
    {
        std::vector<double> transformedData(data.size());
        std::transform(data.begin(), data.end(), transformedData.begin(),
                       [&](double val)
                       { return std::log(val); });
        return transformedData;
    }

    std::vector<double> reciprocalTransform(const std::vector<double> &data)
    {
        std::vector<double> transformedData(data.size());
        std::transform(data.begin(), data.end(), transformedData.begin(),
                       [&](double val)
                       { return 1.0 / val; });
        return transformedData;
    }

    std::vector<double> squareRootTransform(const std::vector<double> &data)
    {
        std::vector<double> transformedData(data.size());
        std::transform(data.begin(), data.end(), transformedData.begin(),
                       [&](double val)
                       { return std::sqrt(val); });
        return transformedData;
    };

    std::vector<double> boxCoxTransform(const std::vector<double> &data, double lambda)
    {
        std::vector<double> transformedData(data.size());

        if (std::abs(lambda) < 1e-6)
        {
            // Handle special case when lambda is close to zero
            std::transform(data.begin(), data.end(), transformedData.begin(),
                           [](double val)
                           { return std::log(val); });
        }
        else
        {
            // General Box-Cox transformation
            std::transform(data.begin(), data.end(), transformedData.begin(),
                           [lambda](double val)
                           {
                               if (val > 0.0)
                               {
                                   return (std::pow(val, lambda) - 1.0) / lambda;
                               }
                               else
                               {
                                   // Handle negative values when lambda is not zero
                                   return -std::pow(-val, lambda);
                               }
                           });
        }

        return transformedData;
    }
};

int main()
{
    MLTransformer transformer;

    // Example usage
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<std::string> categories = {"apple", "banana", "apple", "orange", "banana"};

    std::vector<double> standardizedData = transformer.standardize(data);
    std::vector<double> scaledData = transformer.minMaxScale(data, 0.0, 1.0);
    std::vector<double> hashedData = transformer.featureHashing(categories, 5);
    std::vector<std::vector<double>> polynomialFeatures = transformer.addPolynomialFeatures(data, 3);
    std::vector<std::vector<double>> encodedCategories = transformer.oneHotEncode(categories);

    // Output transformed data
    // (Note: In a real ML scenario, these transformed data would likely be used for further analysis or modeling)
    // (Printing here just for demonstration purposes)
    std::cout << "Standardized Data:";
    for (const auto &val : standardizedData)
    {
        std::cout << " " << val;
    }
    std::cout << std::endl;

    std::cout << "Scaled Data:";
    for (const auto &val : scaledData)
    {
        std::cout << " " << val;
    }
    std::cout << std::endl;

    std::cout << "Feature Hashed Data:";
    for (const auto &val : hashedData)
    {
        std::cout << " " << val;
    }
    std::cout << std::endl;

    std::cout << "Polynomial Features:";
    for (const auto &features : polynomialFeatures)
    {
        for (const auto &val : features)
        {
            std::cout << " " << val;
        }
        std::cout << " |";
    }
    std::cout << std::endl;

    std::cout << "One-Hot Encoded Categories:";
    for (const auto &encodedCategory : encodedCategories)
    {
        for (const auto &val : encodedCategory)
        {
            std::cout << " " << val;
        }
        std::cout << " |";
    }
    std::cout << std::endl;

    data = {1.0, 2.0, 3.0, 4.0, 5.0};
    double lambda = 0.5; // within range (-5, 5)

    std::vector<double> boxCoxTransformedData = transformer.boxCoxTransform(data, lambda);

    // Output transformed data
    std::cout << "Box-Cox Transformed Data:";
    for (const auto &val : boxCoxTransformedData)
    {
        std::cout << " " << val;
    }
    std::cout << std::endl;

    return 0;
}
