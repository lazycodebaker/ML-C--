
#include "include_file.h"

class SimpleImputer
{
private:
    std::vector<std::vector<double>> data;
    std::vector<double> column_means;
    std::vector<double> column_medians;
    std::vector<double> column_most_frequent;
    double fill_value;

public:
    SimpleImputer(std::vector<std::vector<double>> input_data, double fill = 0.0) : data(input_data), fill_value(fill) {}

    void fit()
    {
        int num_columns = data[0].size();
        column_means.resize(num_columns);
        column_medians.resize(num_columns);
        column_most_frequent.resize(num_columns);

        for (int j = 0; j < num_columns; ++j)
        {
            std::vector<double> column_values;
            for (int i = 0; i < data.size(); ++i)
            {
                if (!std::isnan(data[i][j]))
                {
                    column_values.push_back(data[i][j]);
                }
            }
            column_means[j] = calculateMean(column_values);
            column_medians[j] = calculateMedian(column_values);
            column_most_frequent[j] = calculateMostFrequent(column_values);
        }
    }

    std::vector<std::vector<double>> transform(std::string strategy)
    {
        std::vector<std::vector<double>> transformed_data = data;
        for (int i = 0; i < transformed_data.size(); ++i)
        {
            for (int j = 0; j < transformed_data[i].size(); ++j)
            {
                if (std::isnan(transformed_data[i][j]))
                {
                    if (strategy == "mean")
                    {
                        transformed_data[i][j] = column_means[j];
                    }
                    else if (strategy == "median")
                    {
                        transformed_data[i][j] = column_medians[j];
                    }
                    else if (strategy == "most_frequent")
                    {
                        transformed_data[i][j] = column_most_frequent[j];
                    }
                    else if (strategy == "constant")
                    {
                        transformed_data[i][j] = fill_value;
                    }
                }
            }
        }
        return transformed_data;
    }

    double calculateMean(const std::vector<double> &values)
    {
        return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    }

    double calculateMedian(std::vector<double> &values)
    {
        // if the size of the vector is odd then return the middle element
        // else return the average of the two middle elements after sorting the vector
        std::sort(values.begin(), values.end());
        
        return (values.size() % 2 == 0) ? 
            (values[values.size() / 2 - 1] + values[values.size() / 2]) / 2 : 
                values[values.size() / 2];
    }

    double calculateMostFrequent(std::vector<double> &values)
    {
        std::sort(values.begin(), values.end());
        double most_frequent = values[0];
        int max_count = 1;
        int current_count = 1;

        for (int i = 1; i < values.size(); ++i)
        {
            if (values[i] == values[i - 1])
            {
                current_count++;
            }
            else
            {
                if (current_count > max_count)
                {
                    max_count = current_count;
                    most_frequent = values[i - 1];
                }
                current_count = 1;
            }
        }
        if (current_count > max_count)
        {
            most_frequent = values[values.size() - 1];
        }
        return most_frequent;
    }
};

int main()
{
    std::vector<std::vector<double>> input_data = {{7, 4, 3}, {4, NAN, 6}, {10, 5, 5}, {8, 4, NAN}};

    SimpleImputer imputer(input_data);
    imputer.fit();

    std::vector<std::vector<double>> transformed_data = imputer.transform("mean");

    for (const auto &row : transformed_data)
    {
        for (const auto &val : row)
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;

    // MOST FREQUENT
    std::vector<std::vector<double>> transformed_data2 = imputer.transform("most_frequent");

    for (const auto &row : transformed_data2)
    {
        for (const auto &val : row)
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
