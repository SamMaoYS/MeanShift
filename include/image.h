//
// Created by sam on 2020-08-02.
//

#ifndef SEGMENTATION_IMAGE_H
#define SEGMENTATION_IMAGE_H


#include <iostream>
#include <opencv2/highgui.hpp>

namespace win {
    inline void destroyAllWindows() {cv::destroyAllWindows();}
    inline void waitKey(int key = 0) {cv::waitKey(key);}
}

class Image{
public:
    Image();
    Image(const cv::Mat &img, const std::string &name = "img", const std::string &ext = ".png");
    ~Image();

    void setName(const std::string &name);
    void setImage(const cv::Mat &img);

    inline std::string getName() const {return name_;}
    inline std::string getExtension() const {return ext_;}
    inline cv::Mat getCVImage() const {return img_;}
    inline cv::Mat &getCVImage() {return img_;}

    Image copy() const;
    inline bool empty() const {return img_.empty();}
    inline int height() const {return img_.rows;}
    inline int width() const {return img_.cols;}
    inline int channels() const {return img_.channels();}
    void show(bool pause = true, bool destroy = false);

private:
    cv::Mat img_;
    std::string name_;
    std::string ext_;
};

class Images {
public:
    Images(std::string name): name_(std::move(name)) {}
    ~Images();

    void addImage(const Image &img);
    void addImages(const Images &imgs);
    Image at(unsigned int idx = 0) const;

    void setName(const std::string &name);
    inline std::string getName() const {return name_;}

    inline unsigned int size() const {return images_.size();}
    inline bool empty() const {return images_.empty();}

private:
    std::vector<Image> images_;
    std::string name_;
};


#endif //SEGMENTATION_IMAGE_H
