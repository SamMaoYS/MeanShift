//
// Created by sam on 2020-08-02.
//

#ifndef SEGMENTATION_UTILS_H
#define SEGMENTATION_UTILS_H

#include <iostream>
#include <opencv2/highgui.hpp>

namespace utils {

    /* brief@ Indicate the start of each process. */
    void processPrint(const std::string &msg);

    /* brief@ Implement matlab function meshgrid with CV mat. */
    /* detail@ e.g. meshGrid((1:3), (1,2), x, y) result in
             | 1 2 3 |			| 1 1 1 |
        x as | 1 2 3 | and y as | 2 2 2 | */
    void meshGrid(const cv::Range & x_range, const cv::Range & y_range, cv::Mat1i & x_out, cv::Mat1i & y_out);
};


#endif //SEGMENTATION_UTILS_H
