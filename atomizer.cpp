#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <string>

using namespace std;
using namespace cv;


/*
 * auxiliary functions
 */


void rename(Mat &img,
            string &output,
            int &id
) {
    if (id < 10)
        imwrite(output + "0000" + to_string(id) + ".png", img);
    else if (id < 100)
        imwrite(output + "000" + to_string(id) + ".png", img);
    else if (id < 1000)
        imwrite(output + "00" + to_string(id) + ".png", img);
    else if (id < 10000)
        imwrite(output + "0" + to_string(id) + ".png", img);
    else
        imwrite(output + to_string(id) + ".png", img);
}


/*
 * process for atomizing
 */


void atomizer(string &input,
              string &output,
              vector<double> center = {0.25, 0.5},
              double transpancy = 0.75,
              vector<int> out_size = {256, 256},
              double alpha = 0.06,
              double beta = 0.2,
              double gamma = -15
) {
    vector<String> imgs;
    glob(input, imgs);
    Mat frame, image;
    int num = imgs.size();
    pair<double, double> C{out_size[0] * center[0], out_size[1] * center[1]};
    double xi;
    for (int i = 0; i < num; ++i) {
        image = imread(imgs[i]);
        resize(image, frame, Size(out_size[1], out_size[0]));
        for (int j = 0; j < frame.rows; ++j) {
            for (int k = 0; k < frame.cols; ++k) {
                xi = exp(beta * (alpha * sqrt(pow((j - C.first), 2) + pow((k - C.second), 2)) + gamma));
                frame.at<Vec3b>(j, k)[0] = frame.at<Vec3b>(j, k)[0] * xi + transpancy * (1 - xi) * 255;
                frame.at<Vec3b>(j, k)[1] = frame.at<Vec3b>(j, k)[1] * xi + transpancy * (1 - xi) * 255;
                frame.at<Vec3b>(j, k)[2] = frame.at<Vec3b>(j, k)[2] * xi + transpancy * (1 - xi) * 255;
            }
        }
        if (access(output.c_str(), F_OK) == -1) {
            string command = "mkdir -p " + output;
            system(command.c_str());
        }
        rename(frame, output, i+1);
    }
}


int main() {
    string output = "path to output";
    string input = "path to input";
    atomizer(input, output); // please set optional parameters if necessary
}
