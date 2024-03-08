

#include "iostream"
#include "fstream"
#include "vector"
#include "opencv2/opencv.hpp"

// 28 * 28
std::vector<std::vector<unsigned char>> readImages(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    char magicNumber[4];
    char numOfImages[4];
    char numOfRows[4];
    char numOfColumns[4];

    file.read(magicNumber, 4);
    file.read(numOfImages, 4);
    file.read(numOfRows, 4);
    file.read(numOfColumns, 4);

    std::cout << (int)static_cast<unsigned char>(numOfImages[0]) << " -- " << (int)static_cast<unsigned char>(numOfImages[1]) << " -- " << (int)static_cast<unsigned char>(numOfImages[2]) << " -- " << (int)static_cast<unsigned char>(numOfImages[3]) << std::endl;

    int numImages = (static_cast<unsigned char>(numOfImages[0] << 24) | (static_cast<unsigned char>(numOfImages[1] << 16) | (static_cast<unsigned char>(numOfImages[2] << 8) | static_cast<unsigned char>(numOfImages[3]))));
    int numRows = (static_cast<unsigned char>(numOfRows[0] << 24) | (static_cast<unsigned char>(numOfRows[1] << 16) | (static_cast<unsigned char>(numOfRows[2] << 8) | static_cast<unsigned char>(numOfRows[3]))));
    int numCols = (static_cast<unsigned char>(numOfColumns[0] << 24) | (static_cast<unsigned char>(numOfColumns[1] << 16) | (static_cast<unsigned char>(numOfColumns[2] << 8) | static_cast<unsigned char>(numOfColumns[3]))));

    std::cout << "Reading Image File" << std::endl;

    std::cout << "Number of images: " << numImages << std::endl;
    std::cout << "Number of rows: " << numRows << std::endl;
    std::cout << "Number of columns: " << numCols << std::endl;

    std::vector<std::vector<unsigned char>> result;

    for (int i = 0; i < numImages; i++)
    {
        std::vector<unsigned char> image(numRows * numCols);
        file.read(reinterpret_cast<char *>(image.data()), numRows * numCols);
        result.push_back(image);
    }

    file.close();

    return result;
}

std::vector<std::vector<unsigned char>> readLabels(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    char magicNumber[4];
    char numOfImages[4];

    file.read(magicNumber, 4);
    file.read(numOfImages, 4);

    std::cout << (int)static_cast<unsigned char>(numOfImages[0]) << " -- " << (int)static_cast<unsigned char>(numOfImages[1]) << " -- " << (int)static_cast<unsigned char>(numOfImages[2]) << " -- " << (int)static_cast<unsigned char>(numOfImages[3]) << std::endl;

    int numImages = (static_cast<unsigned char>(numOfImages[0] << 24) | (static_cast<unsigned char>(numOfImages[1] << 16) | (static_cast<unsigned char>(numOfImages[2] << 8) | static_cast<unsigned char>(numOfImages[3]))));

    std::cout << "Reading Label File" << std::endl;

    std::cout << "Number of images: " << numImages << std::endl;

    std::vector<std::vector<unsigned char>> result;

    for (int i = 0; i < numImages; i++)
    {
        std::vector<unsigned char> image(1);
        file.read(reinterpret_cast<char *>(image.data()), 1);
        result.push_back(image);
    }

    file.close();

    return result;
}

void load_data_train_model_save()
{
    std::string trainImagesPath = "/Users/anshumantiwari/Documents/CODES/ALGO & ML/C++/ML/MNIST/dataset/train/train-images.idx3-ubyte";
    std::vector<std::vector<unsigned char>> images = readImages(trainImagesPath);

    std::string trainImagesLabelsPath = "/Users/anshumantiwari/Documents/CODES/ALGO & ML/C++/ML/MNIST/dataset/train/train-labels.idx1-ubyte";
    std::vector<std::vector<unsigned char>> labels = readLabels(trainImagesLabelsPath);

    std::vector<cv::Mat> imagesData;
    std::vector<int> labelsData;

    // cv::namedWindow("OPENCV", cv::WINDOW_AUTOSIZE);

    for (int i = 0; i < (int)images.size(); i++)
    {
        cv::Mat tempImg = cv::Mat::zeros(cv::Size(28, 28), CV_8UC1);

        int rowCounter = 0;
        int colCounter = 0;

        for (int j = 0; j < (int)images[i].size(); j++)
        {
            tempImg.at<uchar>(cv::Point(colCounter++, rowCounter)) = (int)images[i][j];

            if (colCounter % 28 == 0)
            {
                rowCounter++;
                colCounter = 0;
            }
        };

        // printing the label
        // std::cout << "Label : " << std::endl;
        // std::cout << (int)labels[i][0] << std::endl;

        imagesData.push_back(tempImg);
        labelsData.push_back((int)labels[i][0]);

        /*
            cv::imshow("OPENCV", tempImg);

            if (cv::waitKey(0) == 'q')
            {
                break;
            }
        */

        std::string testImagesPath = "/Users/anshumantiwari/Documents/CODES/ALGO & ML/C++/ML/MNIST/test_images";
        cv::imwrite(testImagesPath + "/" + std::to_string((int)labels[i][0]) + ".jpg", tempImg);
    }

    std::cout << "Images Size : " << imagesData.size() << std::endl;
    std::cout << "Labels Size : " << labelsData.size() << std::endl;

    // Implementation of MLP
    cv::Ptr<cv::ml::ANN_MLP> ann = cv::ml::ANN_MLP::create();
    ann->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);

    int inputLayersSize = imagesData[0].total();
    int hiddenLayers = 100;
    int outputLayerSize = 10;

    cv::Mat layers = (cv::Mat_<int>(3, 1) << inputLayersSize, hiddenLayers, outputLayerSize);
    ann->setLayerSizes(layers);

    // Preparing the training datasets
    int numberOfSamples = imagesData.size();

    cv::Mat trainingData(numberOfSamples, inputLayersSize, CV_32F);
    cv::Mat labelData(numberOfSamples, outputLayerSize, CV_32F);

    for (int i = 0; i < numberOfSamples; i++)
    {
        cv::Mat image = imagesData[i].reshape(1, 1);
        image.convertTo(trainingData.row(i), CV_32F);

        cv::Mat label = cv::Mat::zeros(1, outputLayerSize, CV_32F);
        label.at<float>((int)labelsData[i]) = 1.0f;
        label.copyTo(labelData.row(i));
    }

    std::cout << "Training Data Size : " << trainingData.size() << std::endl;
    std::cout << "Label Data Size : " << labelData.size() << std::endl;

    // Train

    cv::TermCriteria termCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 10000, 0.001);

    ann->setTermCriteria(termCriteria);
    ann->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.001, 0.1);
    ann->train(trainingData, cv::ml::ROW_SAMPLE, labelData);

    // Save
    ann->save("mnist_trained_model.xml");
}

void load_model_predict_test_image()
{
    cv::Ptr<cv::ml::ANN_MLP> loadedModel = cv::ml::ANN_MLP::load("mnist_trained_model.xml");

    std::string testImageFolderPath = "/Users/anshumantiwari/Documents/CODES/ALGO & ML/C++/ML/MNIST/test_images";

    // iterate all the image file in this folder
    for (const auto &entry : std::filesystem::directory_iterator(testImageFolderPath))
    {
        std::cout << "---------------------- \n"
                  << std::endl;

        std::cout << "Prediction for : " << entry.path().string() << std::endl;
        // iterate for all the images in the folder
        cv::Mat testImage = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);

        cv::resize(testImage, testImage, cv::Size(28, 28));

        // flatten the image
        cv::Mat testImageFlatten = testImage.reshape(1, 1);
        cv::Mat input;

        testImageFlatten.convertTo(input, CV_32F);

        // perform the prediction using the loaded model
        cv::Mat output;
        loadedModel->predict(input, output);

        // std::cout << "Output : " << output << std::endl;

        // finding the class with the highest probability
        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(output, nullptr, &confidence, nullptr, &classIdPoint);

        int predictedClass = classIdPoint.x;
        std::cout << "Predicted Class : " << predictedClass << std::endl;
        std::cout << "Confidence : " << confidence << std::endl;
        std::cout << "---------------------- \n"
                  << std::endl;
    }
}

int main()
{
    // load_data_train_model_save();
    load_model_predict_test_image();
    return 0;
}
