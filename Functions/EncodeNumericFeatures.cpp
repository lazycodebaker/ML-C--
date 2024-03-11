
// converting numeric features to categorical features -- binning (discretization) & binarization

/*
Binning --
    supervised binning --
        1. equal width binning ( width = (max-min)/n ) (uniform binning) ( n = bins )
        2. equal frequency binning ( quantile binning)
        3. k-means binning
    unsupervised binning --
        1. clustering based binning
        2. decision tree based binning
        3. quantile based binning
    custom binning --
        1. binning based on domain knowledge
        2. binning based on business rules
        3. binning based on data distribution
*/

#include "include_file.h"

template <typename T>
class UniformBinning
{
private:
    std::vector<T> data;
    size_t numBins;
    std::vector<T> binEdges;
    std::unordered_map<size_t, size_t> binCounts;

public:
    /// @brief
    /// @param input_data
    /// @param num_bins
    UniformBinning(const std::vector<T> &input_data, size_t num_bins) : data(input_data), numBins(num_bins)
    {
        if (data.empty())
        {
            throw std::invalid_argument("Input data is empty");
        }
        if (numBins < 1)
        {
            throw std::invalid_argument("Number of bins must be at least 1");
        }

        // Sort the data
        std::sort(data.begin(), data.end());

        // Compute bin edges
        size_t dataSize = data.size();
        size_t binSize = dataSize / numBins;

        binEdges.resize(numBins + 1);

        for (size_t i = 0; i <= numBins; ++i)
        {
            size_t index = i * binSize;
            if (index < dataSize)
            {
                binEdges[i] = data[index];
            }
            else
            {
                binEdges[i] = data.back();
            }
        }

        // Count data points in each bin
        binCounts.reserve(numBins);

        std::for_each(data.begin(), data.end(),
                      [&](const T &val)
                      {
                          size_t binIndex = std::upper_bound(binEdges.begin(), binEdges.end(), val) - binEdges.begin();
                          if (binIndex == numBins)
                              binIndex--;
                          binCounts[binIndex]++;
                      });
    }

    size_t getBinCount(size_t binIndex) const
    {
        if (binIndex >= numBins)
        {
            throw std::out_of_range("Bin index out of range");
        }
        return binCounts.at(binIndex);
    }

    const std::vector<T> &getBinEdges() const
    {
        return binEdges;
    }
};

void perform_uniform_binning()
{
    try
    {
        std::vector<double> data = {43.0, 44.0, 15.0, 30.0, 35.0, 2.0, 18.0, 1.0, 19.0, 36.0};
        size_t numBins = 3;

        UniformBinning<double> binning(data, numBins);

        // Print bin edges
        std::cout << "Bin Edges: ";
        for (const auto &edge : binning.getBinEdges())
        {
            std::cout << edge << " ";
        }
        std::cout << std::endl;

        // Print bin counts
        // Print bin counts
        for (size_t i = 0; i < numBins; ++i)
        { // Change here
            try
            {
                std::cout << "Bin " << i << " count: " << binning.getBinCount(i) << std::endl;
            }
            catch (const std::out_of_range &oor)
            {
                std::cerr << "Out of Range error: " << oor.what() << " for Bin " << i << std::endl;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return;
    }
}

// Quantile Binning
template <typename T>
class QuantileBinning
{
private:
    std::vector<T> data;
    size_t numBins;
    std::vector<T> binEdges;
    std::unordered_map<size_t, size_t> binCounts;

public:
    QuantileBinning(const std::vector<T> &input_data, size_t num_bins)
        : data(input_data), numBins(num_bins)
    {

        if (data.empty())
        {
            throw std::invalid_argument("Input data is empty");
        }
        if (numBins < 1)
        {
            throw std::invalid_argument("Number of bins must be at least 1");
        }

        // Sort the data
        std::sort(data.begin(), data.end());

        // Compute bin edges
        size_t dataSize = data.size();
        size_t binSize = dataSize / numBins;
        size_t remainder = dataSize % numBins;

        binEdges.resize(numBins + 1);

        size_t dataIndex = 0;
        for (size_t i = 0; i < numBins; ++i)
        {
            size_t binEnd = dataIndex + binSize + (i < remainder ? 1 : 0);
            binEdges[i] = data[std::min(binEnd, dataSize) - 1];
            dataIndex = std::min(binEnd, dataSize);
        }
        binEdges[numBins] = data.back(); // Last edge

        // Count data points in each bin
        binCounts.reserve(numBins);
        size_t binIndex = 0;
        for (size_t i = 0; i < dataSize; ++i)
        {
            if (data[i] > binEdges[binIndex + 1])
            {
                ++binIndex;
                if (binIndex >= numBins)
                    break;
            }
            binCounts[binIndex]++;
        }
    }

    size_t getBinCount(size_t binIndex) const
    {
        if (binIndex >= numBins)
        {
            throw std::out_of_range("Bin index out of range");
        }
        return binCounts.at(binIndex);
    }

    const std::vector<T> &getBinEdges() const
    {
        return binEdges;
    }
};

void perform_quantile()
{
    std::vector<double> data = {43.0, 44.0, 15.0, 30.0, 35.0, 2.0, 18.0, 1.0, 19.0, 36.0};
    size_t numBins = 3;

    try
    {

        QuantileBinning<double> binning(data, numBins);

        // Print bin edges
        std::cout << "Bin Edges: ";
        for (const auto &edge : binning.getBinEdges())
        {
            std::cout << edge << " ";
        }
        std::cout << std::endl;

        // Print bin counts
        for (size_t i = 0; i < numBins; ++i)
        {
            try
            {
                std::cout << "Bin " << i << " count: " << binning.getBinCount(i) << std::endl;
            }
            catch (const std::out_of_range &oor)
            {
                std::cerr << "Out of Range error: " << oor.what() << " for Bin " << i << std::endl;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return;
    }
};

// K-Means Binning
template <typename T>
class KMeansBinning
{
private:
    std::vector<T> data;
    size_t numBins;
    std::vector<T> binEdges;
    std::vector<size_t> assignments;
    std::vector<T> centroids;
    std::vector<T> prevCentroids;

public:
    KMeansBinning(const std::vector<T> &input_data, size_t num_bins)
        : data(input_data), numBins(num_bins)
    {

        if (data.empty())
        {
            throw std::invalid_argument("Input data is empty");
        }
        if (numBins < 1)
        {
            throw std::invalid_argument("Number of bins must be at least 1");
        }

        // Initialize centroids randomly
        initializeCentroids();

        size_t maxIterations = 1000; // Maximum number of iterations
        size_t iter = 0;
        while (iter < maxIterations && !hasConverged())
        {
            assignDataPoints();
            updateCentroids();
            ++iter;
        }

        computeBinEdges();
    }

    // Initialize centroids randomly
    void initializeCentroids()
    {
        centroids.resize(numBins);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dis(0, data.size() - 1);

        for (size_t i = 0; i < numBins; ++i)
        {
            centroids[i] = data[dis(gen)];
        }
    }

    // Assign data points to nearest centroids
    void assignDataPoints()
    {
        assignments.clear();
        assignments.resize(data.size());

        for (size_t i = 0; i < data.size(); ++i)
        {
            T point = data[i];
            size_t nearestCentroid = 0;
            T minDistance = std::numeric_limits<T>::max();
            for (size_t j = 0; j < numBins; ++j)
            {
                T distance = std::abs(point - centroids[j]);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    nearestCentroid = j;
                }
            }
            assignments[i] = nearestCentroid;
        }
    }

    // Update centroids based on assigned data points
    void updateCentroids()
    {
        prevCentroids = centroids;
        std::unordered_map<size_t, std::pair<T, size_t>> sums;

        for (size_t i = 0; i < data.size(); ++i)
        {
            size_t cluster = assignments[i];
            sums[cluster].first += data[i];
            sums[cluster].second++;
        }

        for (size_t i = 0; i < numBins; ++i)
        {
            centroids[i] = sums[i].first / sums[i].second;
        }
    }

    // Cut function similar to pd.cut
    std::vector<size_t> cut(const std::vector<T> &input)
    {
        std::vector<size_t> result;
        for (const auto &value : input)
        {
            size_t bin_index = 0;
            for (size_t i = 0; i < binEdges.size() - 1; ++i)
            {
                if (value >= binEdges[i] && value < binEdges[i + 1])
                {
                    bin_index = i;
                    break;
                }
            }
            result.push_back(bin_index);
        }
        return result;
    }

    // Check convergence by comparing centroids
    bool hasConverged()
    {
        return prevCentroids == centroids;
    }

    // Compute bin edges based on centroids
    void computeBinEdges()
    {
        binEdges.clear();
        binEdges.resize(numBins + 1);
        std::sort(centroids.begin(), centroids.end());
        for (size_t i = 0; i < numBins; ++i)
        {
            binEdges[i] = (centroids[i] + centroids[i + 1]) / 2;
        }
        binEdges[numBins] = std::numeric_limits<T>::max(); // Last edge
    }

    const std::vector<T> &getBinEdges() const
    {
        return binEdges;
    }
};

void perform_kmeans()
{
    std::vector<double> data = {43.0, 44.0, 15.0, 30.0, 35.0, 2.0, 18.0, 1.0, 19.0, 36.0};
    size_t numBins = 5;

    try
    {
        KMeansBinning<double> binning(data, numBins);

        // Print bin edges
        std::cout << "Bin Edges: ";
        for (const auto &edge : binning.getBinEdges())
        {
            std::cout << edge << " ";
        }
        std::cout << std::endl;

        std::cout << "Cut: ";
        for (const auto &bin_index : binning.cut(data))
        {
            std::cout << bin_index << " ";
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

int main()
{
    std::cout << "K-Means Binning" << std::endl;
    perform_kmeans();
}