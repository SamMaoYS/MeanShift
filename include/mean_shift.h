//
// Created by sam on 2020-08-02.
//

#ifndef SEGMENTATION_MEAN_SHIFT_H
#define SEGMENTATION_MEAN_SHIFT_H


#include "image.h"

class MeanShift {
public:
    MeanShift() = default;
    MeanShift(const Image& img, float hs, float hr);
    ~MeanShift() = default;

    void filter();

    void setImage(const Image& img);
    void setSpatialBandwidth(float hs);
    void setRangeBandwidth(float hr);
protected:
    enum Status{SUCCESS, FAIL_IMAGE, FAIL_HS, FAIL_HR};
    int configure();

private:
    // copy of input image
    Image image_;

    // bandwidths
    // spatial bandwidth
    float hs_;
    // range bandwidth
    float hr_;
};


#endif //SEGMENTATION_MEAN_SHIFT_H
