//
// Created by sam on 2020-08-02.
//

#ifndef SEGMENTATION_MEAN_SHIFT_H
#define SEGMENTATION_MEAN_SHIFT_H


#include "image.h"

class MeanShift {
public:
    MeanShift();
    MeanShift(const Image& img, float hs = 8, float hr = 16, int min_size = 100);
    ~MeanShift();

    void filter();
    void segment();

    void setImage(const Image& img);
    void setSpatialBandwidth(float hs);
    void setRangeBandwidth(float hr);
    void setMinSize(int min_size);

    Image getFilteredImage() const;
    Image getSegmentedImage() const;
    Image getRandomColorImage() const;
    inline int getNumSegments() const {return num_segms_;}
protected:
    enum Status{SUCCESS, FAIL_IMAGE, FAIL_HS, FAIL_HR, FAIL_MIN_SIZE};
    int configure();
    void filterRGB();
    void cluster();
    void merge();
    void mergeRange();
    void mergeMinSize();

    void genSegmentedImage();
    float dist3D2(const cv::Point3f &x, const cv::Point3f &y);
    cv::Vec3b randomColor() const;

    template<typename T>
    void initMerge(T *&adj_nodes, T *&adj_pool);

    template<typename T>
    int getParent(T* adj_nodes, int idx);

    template<typename T>
    void reLabel(T *adj_nodes);
private:
    // copy of input image
    Image image_;
    Image filtered_;
    Image segmented_;

    // bandwidths
    // spatial bandwidth
    float hs_;
    // range bandwidth
    float hr_;
    int min_size_;

    cv::Mat l_filtered_;
    cv::Mat u_filtered_;
    cv::Mat v_filtered_;

    std::vector<cv::Point3f> modes_range_;
    std::vector<cv::Point2i> modes_pos_;
    std::vector<int> modes_size_;

    cv::Mat labels_;

    int num_segms_;
};


#endif //SEGMENTATION_MEAN_SHIFT_H
