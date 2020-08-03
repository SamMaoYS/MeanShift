//
// Created by sam on 2020-08-02.
//

#include "mean_shift.h"
#include "utils.h"
#include <opencv2/imgproc.hpp>

using namespace std;

MeanShift::MeanShift(const Image &img, float hs, float hr) {
    this->image_ = img.copy();
    this->setSpatialBandwidth(hs);
    this->setRangeBandwidth(hr);
}

void MeanShift::setImage(const Image &img) {
    if (img.empty()) {
        cerr << "Input image " << img.getName() << " is empty" << endl;
        return;
    }
    image_ = img.copy();
}

void MeanShift::setSpatialBandwidth(float hs) {
    if (hs <= 0) {
        cerr << "Input spatial bandwidth should be positive" << endl;
        return;
    }
    hs_ = hs;
}

void MeanShift::setRangeBandwidth(float hr) {
    if (hr <= 0) {
        cerr << "Input range bandwidth should be positive" << endl;
        return;
    }
    hr_ = hr;
}

void MeanShift::filter() {
    utils::processPrint("Mean Shift Filter Starts");

    if (configure() != SUCCESS)
        return;

    int height = image_.height();
    int width = image_.width();
    int channels = image_.channels();

    if (channels != 3) {
        cerr << "Currently only support rgb image" << endl;
        return;
    }

    // x is the column index and y is the row index of the image
    cv::Mat1i x_idx, y_idx;
    utils::meshGrid(cv::Range(0, width-1), cv::Range(0, height-1), x_idx, y_idx);

    // convert RGB color space to Luv
    cv::Mat range = image_.getCVImage();
    cv::cvtColor(range, range, cv::COLOR_BGR2Luv);
    vector<cv::Mat> luv;
    cv::split(range, luv);
}

int MeanShift::configure() {
    if (image_.empty()) {
        cerr << "Input image " << image_.getName() << " is empty" << endl;
        return FAIL_IMAGE;
    }
    if (hs_ <= 0) {
        cerr << "Input spatial bandwidth should be positive" << endl;
        return FAIL_HS;
    }
    if (hr_ <= 0) {
        cerr << "Input range bandwidth should be positive" << endl;
        return FAIL_HR;
    }
    return SUCCESS;
}


