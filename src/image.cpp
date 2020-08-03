//
// Created by sam on 2020-08-02.
//

#include "image.h"

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
    img_ = img;
}

void Image::show(bool pause, bool destroy) {
    if (img_.empty()) {
        cerr << "Image " << name_ << " is empty" << endl;
        return;
    }

    cv::namedWindow(name_, cv::WINDOW_AUTOSIZE);
    cv::imshow(name_, img_);
    if (pause)
        cv::waitKey(0);
    if (destroy)
        cv::destroyWindow(name_);
}

Images::~Images() {
    this->images_.clear();
}

void Images::addImage(const Image &img) {
    if (img.getCVImage().empty()) {
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
