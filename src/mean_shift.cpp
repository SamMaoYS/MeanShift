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

    filterRGB();
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

void MeanShift::filterRGB() {
    int height = image_.height();
    int width = image_.width();
    int channels = image_.channels();

    // convert RGB color space to Luv
    cv::Mat range = image_.getCVImage();
    if (channels == 1) {
        // if image is grayscale copy each channel to make a rgb image
        cv::cvtColor(range, range, cv::COLOR_GRAY2BGR);
    }
    cv::cvtColor(range, range, cv::COLOR_BGR2Luv);
    vector<cv::Mat> luv;
    cv::split(range, luv);

    cv::Mat l_filter, u_filter, v_filter;
    l_filter = cv::Mat(height, width, CV_8U);
    u_filter = cv::Mat(height, width, CV_8U);
    v_filter = cv::Mat(height, width, CV_8U);

    int rad_s2 = (int)(hs_*hs_);
    float rad_r2 = 3*hr_*hr_;

    uchar * l_ptr = luv[0].ptr();
    uchar * u_ptr = luv[1].ptr();
    uchar * v_ptr = luv[2].ptr();

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int pt_idx = i*width+j;
            int x_old = j;
            int y_old = i;
            float l_old = (float)l_ptr[pt_idx];
            float u_old = (float)u_ptr[pt_idx];
            float v_old = (float)v_ptr[pt_idx];
            int x_new = 0;
            int y_new = 0;
            float l_new = 0;
            float u_new = 0;
            float v_new = 0;
            float shift = FLT_MAX;

            int iter = 0;
            while (iter < 100 && shift > 1) {
                x_new = 0;
                y_new = 0;
                l_new = 0;
                u_new = 0;
                v_new = 0;
                float c1 = 0;
                float c2 = 0;

                int x_start = max(0, j-(int)hs_);
                int x_end = min(width, j+(int)hs_+1);
                int x_center = (x_start + x_end)/2;
                int y_start = max(0, i-(int)hs_);
                int y_end = min(height, i+(int)hs_+1);
                int y_center = (y_start + y_end)/2;
                for (int y = y_start; y < y_end; ++y) {
                    for (int x = x_start; x < x_end; ++x) {
                        int pt_win_idx = y*width+x;
                        float l_val = (float)l_ptr[pt_win_idx];
                        float u_val = (float)u_ptr[pt_win_idx];
                        float v_val = (float)v_ptr[pt_win_idx];
                        float dl = l_old - l_val;
                        float du = u_old - u_val;
                        float dv = v_old - v_val;

                        int dx = x - x_center;
                        int dy = y - y_center;
                        int ds = dx*dx + dy*dy;
                        float dr = dl*dl + du*du + dv*dv;
                        if (dr <= rad_r2) {
                            float g1 = (float)std::exp(-(float)ds/(2.0*rad_s2));
                            float g2 = (float)std::exp(-(float)dr/(2.0*rad_r2));
                            x_new += (int)(g1*x);
                            y_new += (int)(g1*y);
                            l_new += g2*l_val;
                            u_new += g2*u_val;
                            v_new += g2*v_val;
                            c1 += g1;
                            c2 += g2;
                        }
                    }
                }
                if (c1 < std::numeric_limits<float>::min() || c2 == std::numeric_limits<float>::min()) {
                    iter++;
                    continue;
                }

                x_new = x_new / (int)c1;
                y_new = y_new / (int)c1;
                l_new = l_new / (float)c2;
                u_new = u_new / (float)c2;
                v_new = v_new / (float)c2;
                float dx = (float)(x_new - x_old);
                float dy = (float)(y_new - y_old);
                float dl = l_new - l_old;
                float du = u_new - u_old;
                float dv = v_new - v_old;
                shift = dx*dx + dy*dy + dl*dl + du*du + dv*dv;
                x_old = x_new;
                y_old = y_new;
                l_old = l_new;
                u_old = u_new;
                v_old = v_new;
                iter++;
            }
            l_filter.ptr<uchar>()[pt_idx] = (uchar)l_old;
            u_filter.ptr<uchar>()[pt_idx] = (uchar)u_old;
            v_filter.ptr<uchar>()[pt_idx] = (uchar)v_old;
        }
    }

    cv::Mat tmp1;
    vector<cv::Mat> result;
    result.push_back(l_filter);
    result.push_back(u_filter);
    result.push_back(v_filter);
    cv::merge(result, tmp1);
    cv::cvtColor(tmp1, tmp1, cv::COLOR_Luv2BGR);

    filtered_.setImage(tmp1);
    filtered_.setName(image_.getName()+"_filtered");
}

Image MeanShift::getFilteredImage() const {
    if (filtered_.empty()) {
        cerr << "Filtered image is not generated, please run filter first" << endl;
        return Image();
    }
    return filtered_;
}
