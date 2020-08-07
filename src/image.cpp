//
// Created by sam on 2020-08-02.
//

#include "image.h"
#include <opencv2/imgproc.hpp>

using namespace std;

Image::Image() {
    this->img_.create(0, 0, CV_8U);
    this->name_ = "img";
    this->ext_ = ".png";
}

Image::Image(const cv::Mat &img, const string &name, const string &ext) {
    this->img_ = img.clone();
    this->name_ = name;
    this->ext_ = ext;
}

Image::~Image() {
    this->img_.release();
}

void Image::setName(const string &name) {
    if (name.empty()) {
        cerr << "Input name is empty" << endl;
        return;
    }
    name_ = name;
}

void Image::setImage(const cv::Mat &img) {
    if (img.empty()) {
        cerr << "Input image is empty" << endl;
        return;
    }
    img_ = img.clone();
}

Image Image::copy() const {
    Image tmp;
    tmp.img_ = img_.clone();
    tmp.name_ = name_;
    tmp.ext_ = ext_;
    return tmp;
}

void Image::show(bool pause, bool destroy) {
    if (img_.empty()) {
        cerr << "Image " << name_ << " is empty" << endl;
        return;
    }

    cv::namedWindow(name_, cv::WINDOW_AUTOSIZE);

    int type = img_.type();
    if (type == CV_8UC3 || type == CV_8U) {
        cv::imshow(name_, img_);
    }
    else {
        cv::Mat result;
        cv::Mat depth_8U, mask, depth_8UC3;
        double min, max;
        cv::minMaxLoc(img_, &min, &max);
        float scale = 255.0 / (max - min);
        img_.convertTo(depth_8U, CV_8UC1, scale, -min * scale);
        cv::threshold(depth_8U, mask, 1, 255, cv::THRESH_BINARY);
        mask.convertTo(mask, CV_8UC3);
        cv::applyColorMap(depth_8U, depth_8UC3, cv::COLORMAP_JET);
        cv::bitwise_and(depth_8UC3, depth_8UC3, result, mask);
        cv::resize(result, result, cv::Size(640, 480));
        cv::imshow(name_, result);
    }

    if (pause) {
        // continue until esc key is pressed
        while (cv::waitKey(0) != 27);
    }
    if (destroy)
        cv::destroyWindow(name_);
}

void Image::resize(float scale) {
    if (scale <= 0) {
        cerr << "Input resize scale should be positive" << endl;
        return;
    }
    cv::resize(img_, img_, cv::Size(), scale, scale);
}

Images::~Images() {
    this->images_.clear();
}

void Images::addImage(const Image &img) {
    if (img.empty()) {
        cerr << "Input image " << img.getName() << img.getExtension() << " to Images " << name_ << " is empty" << endl;
        return;
    }
    images_.emplace_back(img);
}

void Images::setName(const string &name) {
    if (name.empty()) {
        cerr << "Input name is empty" << endl;
        return;
    }
    name_ = name;
}

Image Images::at(unsigned int idx) const {
    if (idx >= images_.size()) {
        cerr << "The size of Images " << name_ << " is " << images_.size() << ", the input idx exceeds the maximum index" << endl;
        return Image();
    }
    return images_[idx];
}

void Images::addImages(const Images &imgs) {
    images_.insert(images_.end(), imgs.images_.begin(), imgs.images_.end());
}
