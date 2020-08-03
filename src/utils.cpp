//
// Created by sam on 2020-08-02.
//

#include "utils.h"
#include <iomanip>

using namespace std;

void utils::processPrint(const std::string &msg) {
    int num = 50;
    int num_half_1 = (int)(num - 2 - msg.size())/2;
    int num_half_2 = (msg.size()%2 == 0) ? num_half_1 : num_half_1+1;
    std::string tmp(num, '*');
    cout << tmp << endl;
    cout << "*" << string(num_half_1, ' ') << msg << string(num_half_2, ' ') << "*" << endl;
    cout << tmp << endl;
}

void utils::meshGrid(const cv::Range & x_range, const cv::Range & y_range, cv::Mat1i & x_out, cv::Mat1i & y_out) {
    std::vector<int> x_vec, y_vec;
    for (int i = x_range.start; i <= x_range.end; i++) x_vec.push_back(i);
    for (int i = y_range.start; i <= y_range.end; i++) y_vec.push_back(i);

    cv::Mat x_mat = cv::Mat(x_vec);
    cv::Mat y_mat = cv::Mat(y_vec);

    cv::repeat(x_mat.reshape(1, 1), y_mat.total(), 1, x_out);
    cv::repeat(y_mat.reshape(1, 1).t(), 1, x_mat.total(), y_out);
}

