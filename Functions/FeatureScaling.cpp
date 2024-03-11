
#include "include_file.h"

class FeatureScaler {
private:
    std::vector<double> minValues;
    std::vector<double> maxValues;
    std::vector<double> meanValues;
    std::vector<double> stdDevValues;
    bool isFitted;

public:
    FeatureScaler() : isFitted(false) {}

    void fit(const std::vector<std::vector<double>>& features) {
        size_t numFeatures = features[0].size();
        minValues.resize(numFeatures);
        maxValues.resize(numFeatures);
        meanValues.resize(numFeatures);
        stdDevValues.resize(numFeatures);

        for (size_t i = 0; i < numFeatures; ++i) {
            std::vector<double> column;
            for (const auto& feature : features) {
                column.push_back(feature[i]);
            }

            minValues[i] = *std::min_element(column.begin(), column.end());
            maxValues[i] = *std::max_element(column.begin(), column.end());
            meanValues[i] = std::accumulate(column.begin(), column.end(), 0.0) / column.size();

            double variance = 0.0;
            for (double val : column) {
                variance += (val - meanValues[i]) * (val - meanValues[i]);
            }
            variance /= column.size();
            stdDevValues[i] = std::sqrt(variance);
        }

        isFitted = true;
    }

    std::vector<std::vector<double>> transform(const std::vector<std::vector<double>>& features) {
        if (!isFitted) {
            std::cerr << "Scaler has not been fitted. Call fit method first." << std::endl;
            return {};
        }

        std::vector<std::vector<double>> scaledFeatures(features.size(), std::vector<double>(features[0].size()));

        for (size_t i = 0; i < features.size(); ++i) {
            for (size_t j = 0; j < features[0].size(); ++j) {
                scaledFeatures[i][j] = (features[i][j] - meanValues[j]) / stdDevValues[j];
            }
        }

        return scaledFeatures;
    }
};

int main() {
    // Example usage
    std::vector<std::vector<double>> features = {{1.0, 2.0, 3.0},
                                                 {4.0, 5.0, 6.0},
                                                 {7.0, 8.0, 9.0}};

    FeatureScaler scaler;
    scaler.fit(features);
    std::vector<std::vector<double>> scaledFeatures = scaler.transform(features);

    // Printing scaled features
    for (const auto& feature : scaledFeatures) {
        for (double val : feature) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

