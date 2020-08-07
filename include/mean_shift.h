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

    void filter(int mode = MODE_RGB);
    void segment(int mode = MODE_RGB);

    void setImage(const Image& img);
    void setSpatialBandwidth(float hs);
    void setRangeBandwidth(float hr);
    void setMinSize(int min_size);

    void setIterationThresholds(int max_iter, float min_shift);
    void setMergeMaxAdjacent(int max_adj);
    void setLargeAreaThresholds(float height, float width);

    enum Mode{MODE_RGB, MODE_DEPTH, MODE_NORMAL};
    enum Output{OUT_FILTER = 1u << 0u, OUT_SEGM = 1u << 1u, OUT_RANDOM = 1u << 2u, OUT_MASK = 1u << 3u, OUT_LA = 1u << 4u};
    Images getResultImages(uint8_t code) const;
    inline int getNumSegments() const {return num_segms_;}
protected:
    enum Status{SUCCESS, FAIL_IMAGE, FAIL_HS, FAIL_HR, FAIL_MIN_SIZE, FAIL_MODE};
    int configure();
    void filterRGB();
    void filterDepth();
    void clusterRGB();
    void clusterDepth();
    void mergeRGB();
    void mergeRangeRGB();
    void mergeMinSizeRGB();
    void mergeDepth();
    void mergeRangeDepth();
    void mergeMinSizeDepth();

    cv::Mat gaussianKernel(int size);

    Image getFilteredImage() const;
    Image getSegmentedImage() const;
    Image getSegmentedRGB() const;
    Image getSegmentedDepth() const;
    Image getRandomColorImage() const;
    Images getMaskImages() const;

    Images getLargeAreaMasks() const;
    void computePCA(const cv::Mat &locations, std::vector<cv::Vec2d> &eigen_vecs, std::vector<double> &eigen_val,
                       cv::Point &center_pt) const;

    float dist1D2(float x, float y);
    float dist3D2(const cv::Point3f &x, const cv::Point3f &y);
    cv::Vec3b randomColor() const;

    template<typename T>
    void initMerge(T *&adj_nodes, T *&adj_pool);

    template<typename T>
    int getParent(T* adj_nodes, int idx);

    template<typename T>
    void reLabelRGB(T *adj_nodes);

    template<typename T>
    void reLabelDepth(T *adj_nodes);
private:
    // copy of input image
    Image image_;
    Image filtered_;

    int mode_;

    // bandwidths
    // spatial bandwidth
    float hs_;
    // range bandwidth
    float hr_;
    int min_size_;

    int max_iter_;
    float min_shift_;
    int max_adj_;

    cv::Mat l_filtered_;
    cv::Mat u_filtered_;
    cv::Mat v_filtered_;

    cv::Mat z_filtered_;

    float max_eigen_val1_;
    float max_eigen_val2_;

    std::vector<cv::Point3f> modes_range_;
    std::vector<float> modes_depth_;
    std::vector<cv::Point2i> modes_pos_;
    std::vector<int> modes_size_;

    cv::Mat labels_;

    int num_segms_;
};


#endif //SEGMENTATION_MEAN_SHIFT_H
