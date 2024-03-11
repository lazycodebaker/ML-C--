#include <iostream>
#include <unordered_map>
#include <vector>
#include <stdexcept>

// LABEL ENCODER
template <typename T>
class LabelEncoder
{
private:
    std::unordered_map<T, int> label_to_index;
    std::unordered_map<int, T> index_to_label;
    int next_index = 0;

public:
    void fit(const std::vector<T> &labels)
    {
        for (const auto &label : labels)
        {
            label_to_index.emplace(label, next_index);
            index_to_label.emplace(next_index, label);
            ++next_index;
        }
    }

    auto encode(const std::vector<T> &labels) const
    {
        std::vector<int> encoded_labels;
        for (const auto &label : labels)
        {
            if (auto it = label_to_index.find(label); it != label_to_index.end())
            {
                encoded_labels.push_back(it->second);
            }
            else
            {
                throw std::invalid_argument("Unknown label: " + label);
            }
        }
        return encoded_labels;
    }

    auto decode(const std::vector<int> &encoded_labels) const
    {
        std::vector<T> decoded_labels;
        for (int index : encoded_labels)
        {
            if (auto it = index_to_label.find(index); it != index_to_label.end())
            {
                decoded_labels.push_back(it->second);
            }
            else
            {
                throw std::invalid_argument("Unknown index: " + std::to_string(index));
            }
        }
        return decoded_labels;
    }
};

//  ONE HOT ENCODER WITH MULTI-COLINEARITY CHECK AND DUMMY VARIABLE TRAP CHECK AND DECODER
template <typename T>
class OneHotEncoder
{
private:
    std::unordered_map<std::string, int> feature_to_index;
    std::vector<std::string> index_to_feature;

public:
    void fit(const std::vector<std::string> &features)
    {
        for (const auto &feature : features)
        {
            if (feature_to_index.find(feature) == feature_to_index.end())
            {
                feature_to_index[feature] = feature_to_index.size();
                index_to_feature.push_back(feature);
            }
        }
    }

    auto encode(const std::vector<std::string> &features) const
    {
        std::vector<std::vector<int>> encoded_features;
        for (const auto &feature : features)
        {
            if (feature_to_index.find(feature) == feature_to_index.end())
            {
                throw std::invalid_argument("Unknown feature: " + feature);
            }
            std::vector<int> encoded_feature(index_to_feature.size(), 0);
            encoded_feature[feature_to_index.at(feature)] = 1;
            encoded_features.push_back(encoded_feature);
        }
        return encoded_features;
    }

    auto decode(const std::vector<std::vector<int>> &encoded_features) const
    {
        std::vector<std::string> decoded_features;
        for (const auto &encoded_feature : encoded_features)
        {
            int active_index = -1;
            for (int i = 0; i < encoded_feature.size(); ++i)
            {
                if (encoded_feature[i] == 1)
                {
                    if (active_index != -1)
                    {
                        throw std::invalid_argument("Multi-collinearity detected in one-hot encoding.");
                    }
                    active_index = i;
                }
            }
            if (active_index == -1)
            {
                throw std::invalid_argument("No active feature found in one-hot encoding.");
            }
            decoded_features.push_back(index_to_feature[active_index]);
        }
        return decoded_features;
    }
};

int main()
{
    LabelEncoder<std::string> encoder;
    std::vector<std::string> labels = {"cat", "dog", "mouse", "cat", "dog", "dog"};

    encoder.fit(labels);

    const auto encoded_labels = encoder.encode({"cat", "dog", "dog", "mouse"});
    std::cout << "Encoded labels:";
    for (int label : encoded_labels)
    {
        std::cout << " " << label;
    }
    std::cout << std::endl;

    const auto decoded_labels = encoder.decode({0, 1, 1, 2});
    std::cout << "Decoded labels:";
    for (const auto &label : decoded_labels)
    {
        std::cout << " " << label;
    }
    std::cout << std::endl;

    std::cout << "-------------------" << std::endl;

    std::cout << "One hot encoding:" << std::endl;

    OneHotEncoder<std::string> one_hot_encoder;

    std::vector<std::string> features = {"red", "green", "blue", "red", "green", "green"};
    one_hot_encoder.fit(features);

    const auto encoded_features = one_hot_encoder.encode({"red", "green", "green", "blue"});

    std::cout << "Encoded features:";
    for (const auto& feature : encoded_features) {
        std::cout << " [";
        for (int val : feature) {
            std::cout << val << " ";
        }
        std::cout << "]";
    }
    std::cout << std::endl;

    const auto decoded_features = one_hot_encoder.decode({{1, 0, 0}, {0, 1, 0}, {0, 1, 0}, {0, 0, 1}});
    std::cout << "Decoded features:";
    for (const auto& feature : decoded_features) {
        std::cout << " " << feature;
    }
    std::cout << std::endl;




    return 0;
}
