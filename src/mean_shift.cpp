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
    range.convertTo(range, CV_32F);
    vector<cv::Mat1f> luv;
    cv::split(range, luv);

    int num_pt = width*height;

    cv::Mat l, u, v;
    l = cv::Mat(height, width, CV_32F);
    u = cv::Mat(height, width, CV_32F);
    v = cv::Mat(height, width, CV_32F);

    int rad_s = (int)hs_;
    int rad_s2 = rad_s*rad_s;
    float rad_r = hr_;
    float rad_r2 = 3*rad_r*rad_r;
    for (int i = 0; i < num_pt; ++i) {
        float x_old = x_idx.ptr<int>()[i];
        float y_old = y_idx.ptr<int>()[i];
        float l_old = luv[0].ptr<float>()[i];
        float u_old = luv[1].ptr<float>()[i];
        float v_old = luv[2].ptr<float>()[i];
        float x_new = 0;
        float y_new = 0;
        float l_new = 0;
        float u_new = 0;
        float v_new = 0;
        float shift = FLT_MAX;

        int iter = 0;
        while (iter < 10 && shift > 3) {
            x_new = 0;
            y_new = 0;
            l_new = 0;
            u_new = 0;
            v_new = 0;

            float c1 = 0;
            float c2 = 0;
            for (int j = -rad_s; j < rad_s; ++j) {
                int y = (int)y_old+j;
                for (int k = -rad_s; k < rad_s; ++k) {
                    int x = (int)x_old+k;
                    if (x >= 0 && x < width && y >= 0 && y < height) {
                        int ds = j*j + k*k;
                        if (ds <= rad_s2) {
                            int idx_c = y*width+x;
                            float dl = l_old - luv[0].ptr<float>()[idx_c];
                            float du = u_old - luv[1].ptr<float>()[idx_c];
                            float dv = v_old - luv[2].ptr<float>()[idx_c];
                            float dr = dl*dl + du*du + dv*dv;
                            if ( dr <= rad_r2) {
                                float g1 = (float)std::exp(-(float)ds/(2.0*rad_s2));
                                float g2 = (float)std::exp(-(float)dr/(2.0*rad_r2));
                                x_new += g1*(float)x;
                                y_new += g1*(float)y;
                                l_new += g2*luv[0].ptr<float>()[idx_c];
                                u_new += g2*luv[1].ptr<float>()[idx_c];
                                v_new += g2*luv[2].ptr<float>()[idx_c];
                                c1 += g1;
                                c2 += g2;
                            }
                        }
                    }
                }
            }
            x_new = x_new / c1;
            y_new = y_new / c1;
            l_new = l_new / c2;
            u_new = u_new / c2;
            v_new = v_new / c2;
            float dx = x_new - x_old;
            float dy = y_new - y_old;
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
        l.ptr<float>()[i] = l_old;
        u.ptr<float>()[i] = u_old;
        v.ptr<float>()[i] = v_old;
    }
    cv::Mat tmp1;
    vector<cv::Mat> result;
    result.push_back(l);
    result.push_back(u);
    result.push_back(v);
    cv::merge(result, tmp1);
    tmp1.convertTo(tmp1, CV_8UC1);
    cv::cvtColor(tmp1, tmp1, cv::COLOR_Luv2BGR);
    Image t(tmp1);
    t.show();
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


