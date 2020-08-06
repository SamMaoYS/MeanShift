//
// Created by sam on 2020-08-02.
//

#include "mean_shift.h"
#include "utils.h"
#include <opencv2/imgproc.hpp>
#include <stack>

using namespace std;

class AdjNodesImpl {
public:
    AdjNodesImpl();
    ~AdjNodesImpl() = default;

    int label;
    AdjNodesImpl* next;
    bool insert(AdjNodesImpl* node);

private:
    AdjNodesImpl* cur_;
    AdjNodesImpl* prev_;
    bool exist_;
};

AdjNodesImpl::AdjNodesImpl() {
    label = -1;
    next = nullptr;
}

bool AdjNodesImpl::insert(AdjNodesImpl *node) {
    if (!next) {
        next = node;
        node->next = nullptr;
        return true;
    }

    if (next->label > node->label) {
        node->next = next;
        next = node;
        return true;
    }

    cur_ = next;
    while (cur_) {
        if (node->label == cur_->label)
            return false;
        else if (cur_->next == nullptr || cur_->next->label > node->label) {
            node->next = next;
            next = node;
            break;
        }
        cur_ = cur_->next;
    }
    return true;
}

typedef AdjNodesImpl AdjNodes;

MeanShift::MeanShift() {
    this->setSpatialBandwidth(8);
    this->setRangeBandwidth(8);
    this->setMinSize(100);
    this->setIterationThresholds(100, 1);
    this->setMergeMaxAdjacent(10);
}

MeanShift::MeanShift(const Image &img, float hs, float hr, int min_size) {
    this->image_ = img.copy();
    this->setSpatialBandwidth(hs);
    this->setRangeBandwidth(hr);
    this->setMinSize(min_size);
    this->setIterationThresholds(100, 1);
    this->setMergeMaxAdjacent(10);
}

MeanShift::~MeanShift() {
    l_filtered_.release();
    u_filtered_.release();
    v_filtered_.release();

    modes_range_.clear();
    modes_pos_.clear();
    modes_size_.clear();

    labels_.release();
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
    hr_ = sqrt(3)*hr;
}

void MeanShift::setMinSize(int min_size) {
    if (min_size <= 0) {
        cerr << "Input minimum size should be positive" << endl;
        return;
    }
    min_size_ = min_size;
}

void MeanShift::filter() {
    utils::processPrint("Mean Shift Filter Starts");

    if (configure() != SUCCESS)
        return;

    filterRGB();
    cout << "Filter process ends" << endl;
}

void MeanShift::segment() {
    utils::processPrint("Mean Shift Segment Starts");

    if (configure() != SUCCESS)
        return;

    if (filtered_.empty()) {
        filterRGB();
        cout << "Filter process ends" << endl;
    }

    cluster();
    merge();
    modes_range_.resize(num_segms_);
    modes_pos_.resize(num_segms_);
    modes_size_.resize(num_segms_);
    cout << "Segment process ends" << endl;
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
    if (min_size_ <= 0) {
        cerr << "Input minimum size should be positive" << endl;
        return FAIL_MIN_SIZE;
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
    std::vector<cv::Mat> luv;
    cv::split(range, luv);
    range.release();

    l_filtered_.create(height, width, CV_8U);
    u_filtered_.create(height, width, CV_8U);
    v_filtered_.create(height, width, CV_8U);

    float rad_r2 = hr_*hr_;

    uchar * l_ptr = luv[0].ptr();
    uchar * u_ptr = luv[1].ptr();
    uchar * v_ptr = luv[2].ptr();

    cv::Mat hs_kernel = gaussianKernel((int)(2*hs_+1));
    cv::Mat hr_kernel = gaussianKernel((int)(2*hr_+1));

#pragma omp parallel for collapse(2)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int pt_idx = i*width+j;
            float x_old = (float)j, y_old = (float)i;
            float l_old = (float)l_ptr[pt_idx], u_old = (float)u_ptr[pt_idx], v_old = (float)v_ptr[pt_idx];
            float x_new = 0, y_new = 0;
            float l_new = 0, u_new = 0, v_new = 0;
            float shift = FLT_MAX;

            int iter = 0;
            while (iter < max_iter_ && shift > min_shift_) {
                x_new = 0, y_new = 0;
                l_new = 0, u_new = 0, v_new = 0;
                float c1 = 0, c2 = 0;

                int x_start = max(0, j-(int)hs_), x_end = min(width, j+(int)hs_+1);
                int y_start = max(0, i-(int)hs_), y_end = min(height, i+(int)hs_+1);
                for (int y = y_start; y < y_end; ++y) {
                    for (int x = x_start; x < x_end; ++x) {
                        int pt_win_idx = y*width+x;
                        float l_win = l_ptr[pt_win_idx], u_win = u_ptr[pt_win_idx], v_win = v_ptr[pt_win_idx];
                        float dl = l_win-l_old, du = u_win-u_old, dv = v_win-v_old;
                        float dr2 = dl*dl + du*du + dv*dv;
                        if (dr2 <= rad_r2) {
                            float dr = sqrt(dr2);
                            float g1 = hs_kernel.at<float>(x-x_start, y-y_start);
                            float g2 = hr_kernel.at<float>(dr, dr);
                            x_new += g1*x, y_new +=g1*y;
                            l_new += g2*l_win, u_new += g2*u_win, v_new += g2*v_win;
                            c1 += g1;
                            c2 += g2;
                        }
                    }
                }
                if (c1 <= std::numeric_limits<float>::min() || c2 <= std::numeric_limits<float>::min()) {
                    iter++;
                    continue;
                }
                x_new /= c1, y_new /= c1;
                l_new /= c2, u_new /= c2, v_new /= c2;
                float dx = x_new - x_old, dy = y_new - y_old;
                float dl = l_new - l_old, du = u_new - u_old, dv = v_new - v_old;
                shift = dx*dx + dy*dy + dl*dl + du*du + dv*dv;
                x_old = x_new, y_old = y_new;
                l_old = l_new, u_old = u_new, v_old = v_new;
                iter++;
            }
            l_filtered_.ptr<uchar>()[pt_idx] = (uchar)l_old;
            u_filtered_.ptr<uchar>()[pt_idx] = (uchar)u_old;
            v_filtered_.ptr<uchar>()[pt_idx] = (uchar)v_old;
        }
    }
    luv.clear();

    cv::Mat tmp1;
    vector<cv::Mat> result;
    result.emplace_back(l_filtered_);
    result.emplace_back(u_filtered_);
    result.emplace_back(v_filtered_);
    cv::merge(result, tmp1);
    cv::cvtColor(tmp1, tmp1, cv::COLOR_Luv2BGR);

    filtered_.setImage(tmp1);
    filtered_.setName(image_.getName()+"_filtered");
    tmp1.release();
    result.clear();
}

Image MeanShift::getFilteredImage() const {
    if (filtered_.empty()) {
        cerr << "Filtered image is not generated, please run filter first" << endl;
        return Image();
    }
    return filtered_;
}

void MeanShift::cluster() {
    int height = image_.height();
    int width = image_.width();

    num_segms_ = 0;
    int l_idx = -1;

    float rad_r2 = hr_*hr_;

    cv::Mat real_l = l_filtered_.clone();
    real_l.convertTo(real_l, CV_32F);
    cv::Mat real_u = u_filtered_.clone();
    real_u.convertTo(real_u, CV_32F);
    cv::Mat real_v = v_filtered_.clone();
    real_v.convertTo(real_v, CV_32F);

    float * l_ptr = real_l.ptr<float>();
    float * u_ptr = real_u.ptr<float>();
    float * v_ptr = real_v.ptr<float>();

    const cv::Point2i eight_connect[8] = {cv::Point2i(-1, -1), cv::Point2i(-1, 0), cv::Point2i(-1, 1), cv::Point2i(0, -1), cv::Point2i(0, 1), cv::Point2i(1, -1), cv::Point2i(1, 0), cv::Point2i(1, 1)};

    labels_ = cv::Mat(height, width, CV_32S, cv::Scalar::all(-1));
    int* labels_ptr = labels_.ptr<int>();

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int pt_idx = i*width+j;

            if (labels_ptr[pt_idx] < 0) {
                labels_ptr[pt_idx] = ++l_idx;

                cv::Point2i  ref_pt(i, j);
                cv::Point3f pt_luv(l_ptr[pt_idx], u_ptr[pt_idx], v_ptr[pt_idx]);
                modes_range_.emplace_back(pt_luv);
                modes_pos_.emplace_back(ref_pt);
                modes_size_.push_back(1);

                vector<cv::Point2i> neighbors;
                neighbors.emplace_back(ref_pt);

                while (!neighbors.empty()) {
                    cv::Point2i cur_pt = neighbors.back();
                    neighbors.pop_back();

                    for (int k = 0; k < 8; ++k) {
                        cv::Point2i ne_pt = cur_pt + eight_connect[k];
                        if (ne_pt.x >= 0 && ne_pt.y >= 0 && ne_pt.x < height && ne_pt.y < width) {
                            int ne_pt_idx = ne_pt.x*width + ne_pt.y;
                            cv::Point3f ne_pt_luv(l_ptr[ne_pt_idx], u_ptr[ne_pt_idx], v_ptr[ne_pt_idx]);
                            float dr2 = dist3D2(pt_luv, ne_pt_luv);
                            if (dr2 < rad_r2 && labels_ptr[ne_pt_idx] < 0) {
                                labels_ptr[ne_pt_idx] = l_idx;
                                neighbors.emplace_back(ne_pt);
                                modes_range_[l_idx] += ne_pt_luv;
                                modes_pos_[l_idx] += ne_pt;
                                modes_size_[l_idx]++;
                            }
                        }
                    }
                }
                modes_range_[l_idx] /= modes_size_[l_idx];
                modes_pos_[l_idx] /= modes_size_[l_idx];
            }
        }
    }
    num_segms_ = ++l_idx;
}

float MeanShift::dist3D2(const cv::Point3f &x, const cv::Point3f &y) {
    cv::Point3f dist = x-y;
    return dist.dot(dist);
}

Image MeanShift::getSegmentedImage() const {
    int height = image_.height();
    int width = image_.width();
    cv::Mat segmented_cv = cv::Mat(height, width, CV_8UC3, cv::Scalar::all(0));

#pragma omp parallel for collapse(2)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cv::Point3f luv= modes_range_[labels_.at<int>(i,j)];
            cv::Vec3b luv_8U;
            luv_8U[0] = (uchar)luv.x;
            luv_8U[1] = (uchar)luv.y;
            luv_8U[2] = (uchar)luv.z;
            segmented_cv.at<cv::Vec3b>(i,j) = luv_8U;
        }
    }

    cv::cvtColor(segmented_cv, segmented_cv, cv::COLOR_Luv2BGR);
    if (image_.channels() == 1)
        cv::cvtColor(segmented_cv, segmented_cv, cv::COLOR_BGR2GRAY);
    Image segmented(segmented_cv, image_.getName() + "_segmented");
    segmented_cv.release();

    return segmented;
}

cv::Vec3b MeanShift::randomColor() const{
    cv::Vec3b color;
    color[0] = (uchar) rand();
    color[1] = (uchar) rand();
    color[2] = (uchar) rand();
    return color;
}

Image MeanShift::getRandomColorImage() const {
    int height = image_.height();
    int width = image_.width();
    cv::Mat cv_random = cv::Mat(height, width, CV_8UC3, cv::Scalar::all(0));
    cv::Vec3b *colors = new cv::Vec3b[num_segms_];

#pragma omp parallel for
    for (int i = 0; i < num_segms_; i++)
        colors[i] = randomColor();

#pragma omp parallel for collapse(2)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cv_random.at<cv::Vec3b>(i,j) = colors[labels_.at<int>(i,j)];
        }
    }

    Image random(cv_random, image_.getName() + "_random_color");
    cv_random.release();
    return random;
}

void MeanShift::merge() {
    mergeRange();
    mergeMinSize();
}

template<typename T>
int MeanShift::getParent(T *adj_nodes, int idx) {
    while (adj_nodes[idx].label != idx) idx = adj_nodes[idx].label;
    return idx;
}

template<typename T>
void MeanShift::initMerge(T *&adj_nodes, T *&adj_pool) {
    int height = image_.height();
    int width = image_.width();

    adj_nodes = new AdjNodes[num_segms_];
    int adj_pool_size = num_segms_*max_adj_;
    adj_pool = new AdjNodes[adj_pool_size];

#pragma omp parallel for
    for (int i = 0; i < num_segms_; ++i) {
        adj_nodes[i].label = i;
        adj_nodes[i].next = nullptr;
    }

#pragma omp parallel for
    for (int i = 0; i < adj_pool_size-1; ++i)
        adj_pool[i].next = &adj_pool[i+1];
    adj_pool[adj_pool_size-1].next = nullptr;
    AdjNodes *node1, *node2, *free_nodes_old, *free_nodes;
    free_nodes = adj_pool;

    int *labels_ptr = labels_.ptr<int>();
    auto connect_nodes = [&](const int &idx1, const int &idx2) {
        node1 = free_nodes;
        if (node1 == nullptr) return;
        node2 = free_nodes->next;
        if (node2 == nullptr) return;
        free_nodes_old = free_nodes;
        free_nodes = free_nodes->next->next;
        node1->label = labels_ptr[idx1];
        node2->label = labels_ptr[idx2];
        if (adj_nodes[node1->label].insert(node2))
            adj_nodes[node2->label].insert(node1);
        else
            free_nodes = free_nodes_old;
    };

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int pt_idx = i*width + j;
            int top_idx = pt_idx - width;
            int left_idx = pt_idx - 1;
            if (i > 0 && labels_ptr[pt_idx] != labels_ptr[top_idx])
                connect_nodes(pt_idx, top_idx);
            if (j > 0 && labels_ptr[pt_idx] != labels_ptr[left_idx])
                connect_nodes(pt_idx, left_idx);
        }
    }
}

void MeanShift::mergeRange() {
    float rad_r2 = hr_*hr_;

    int num_segms_old = num_segms_;
    for (int iter = 0, delta = 1; iter < 5 && delta > 0; ++iter) {
        AdjNodes *adj_nodes, *adj_pool;

        initMerge<AdjNodes>(adj_nodes, adj_pool);

        if (!adj_nodes || !adj_pool) {
            cerr << "Bad Memory allocation during merge" << endl;
            return;
        }

        for (int i = 0; i < num_segms_; ++i) {
            AdjNodes *neighbor = adj_nodes[i].next;
            while (neighbor) {
                if (dist3D2(modes_range_[i], modes_range_[neighbor->label]) < rad_r2) {
                    int ref_merge = getParent<AdjNodes>(adj_nodes, i);
                    int ne_merge = getParent<AdjNodes>(adj_nodes, neighbor->label);
                    if (ref_merge < ne_merge)
                        adj_nodes[ne_merge].label = ref_merge;
                    else
                        adj_nodes[ref_merge].label = ne_merge;
                }
                neighbor = neighbor->next;
            }
        }

        reLabel<AdjNodes>(adj_nodes);

        delete adj_nodes;
        delete adj_pool;
        delta = num_segms_old - num_segms_;
        num_segms_old = num_segms_;
    }
}

void MeanShift::mergeMinSize() {
    int min_count;
    do {
        min_count = 0;
        AdjNodes *adj_nodes, *adj_pool;

        initMerge<AdjNodes>(adj_nodes, adj_pool);

        if (!adj_nodes || !adj_pool) {
            cerr << "Bad Memory allocation during merge" << endl;
            return;
        }

        for (int i = 0; i < num_segms_; ++i) {
            if (modes_size_[i] < min_size_) {
                ++min_count;
                AdjNodes *neighbor = adj_nodes[i].next;
                if (neighbor == nullptr) {
                    min_count = 0;
                    continue;
                }
                int ne_merge = neighbor->label;
                float min_dist = dist3D2(modes_range_[i], modes_range_[ne_merge]);
                neighbor = neighbor->next;
                while (neighbor) {
                    float tmp = dist3D2(modes_range_[i], modes_range_[neighbor->label]);
                    if (tmp < min_dist) {
                        min_dist = tmp;
                        ne_merge = neighbor->label;
                    }
                    neighbor = neighbor->next;
                }

                int ref_merge = getParent<AdjNodes>(adj_nodes, i);
                ne_merge = getParent<AdjNodes>(adj_nodes, ne_merge);
                if (ref_merge < ne_merge)
                    adj_nodes[ne_merge].label = ref_merge;
                else
                    adj_nodes[ref_merge].label = ne_merge;
            }
        }

        reLabel<AdjNodes>(adj_nodes);

        delete adj_nodes;
        delete adj_pool;
    } while (min_count > 0);
}

template<typename T>
void MeanShift::reLabel(T *adj_nodes) {
    int height = image_.height();
    int width = image_.width();

    for (int i = 0; i < num_segms_; ++i) {
        int ref_merge = getParent<AdjNodes>(adj_nodes, i);
        adj_nodes[i].label = ref_merge;
    }

    int *modes_size_buffer = new int[num_segms_]();
    cv::Point3f *modes_range_buffer = new cv::Point3f[num_segms_];
    cv::Point2i *modes_pos_buffer = new cv::Point2i[num_segms_];
    int *labels_buffer = new int[num_segms_];
    fill(modes_range_buffer, modes_range_buffer+num_segms_, cv::Point3f(0));
    fill(modes_pos_buffer, modes_pos_buffer+num_segms_, cv::Point2i(0));
    fill(labels_buffer, labels_buffer+num_segms_, -1);

    for (int i = 0; i < num_segms_; ++i) {
        int ref_merge = adj_nodes[i].label;
        modes_size_buffer[ref_merge] += modes_size_[i];
        modes_range_buffer[ref_merge] += modes_range_[i]*modes_size_[i];
        modes_pos_buffer[ref_merge] += modes_pos_[i]*modes_size_[i];
    }
    int l_idx = -1;
    for (int i = 0; i < num_segms_; ++i) {
        int ref_merge = adj_nodes[i].label;
        if (labels_buffer[ref_merge] < 0) {
            labels_buffer[ref_merge] = ++l_idx;
            if (modes_size_buffer[ref_merge] == 0)
                continue;
            modes_range_[l_idx] = modes_range_buffer[ref_merge]/modes_size_buffer[ref_merge];
            modes_pos_[l_idx] = modes_pos_buffer[ref_merge]/modes_size_buffer[ref_merge];
            modes_size_[l_idx] = modes_size_buffer[ref_merge];
        }
    }
    num_segms_ = ++l_idx;

    int *labels_ptr = labels_.ptr<int>();
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int pt_idx = i*width + j;
            labels_ptr[pt_idx] = labels_buffer[adj_nodes[labels_ptr[pt_idx]].label];
        }
    }

    delete [] modes_range_buffer;
    delete [] modes_pos_buffer;
    delete [] modes_size_buffer;
    delete [] labels_buffer;
}

Images MeanShift::getResultImages(uint8_t code) const {
    Images results(image_.getName() + "_results");
    if (code & OUT_FILTER) {
        results.addImage(getFilteredImage());
    }
    if (code & OUT_SEGM) {
        results.addImage(getSegmentedImage());
    }
    if (code & OUT_RANDOM) {
        results.addImage(getRandomColorImage());
    }
    if (code & OUT_MASK) {
        results.addImages(getMaskImages());
    }
    return results;
}

Images MeanShift::getMaskImages() const {
    Images masks(image_.getName() + "masks");
    int height = image_.height();
    int width = image_.width();

#pragma omp parallel for
    for (int i = 0; i < num_segms_; ++i) {
        cv::Mat mask(height, width, CV_8U, cv::Scalar::all(0));
        mask.setTo(0, labels_ != i);
        mask.setTo(255, labels_ == i);
#pragma omp critical
        masks.addImage(Image(mask, image_.getName()+"_mask" + to_string(i)));
    }
    return masks;
}

void MeanShift::setIterationThresholds(int max_iter, float min_shift) {
    if (max_iter <= 0 || min_shift <= 0) {
        cerr << "Input iteration thresholds should be positive" << endl;
        return;
    }
    if (max_iter > 200) {
        cerr << "Maximum iteration is too large, could cause process running very slow" << endl;
        return;
    }

    max_iter_ = max_iter;
    min_shift_ = min_shift;
}

void MeanShift::setMergeMaxAdjacent(int max_adj) {
    if (max_adj <= 0) {
        cerr << "Input merge maximum number of adjacent segments should be positive" << endl;
        return;
    }
    if (max_adj > 30) {
        cerr << "Input merge maximum number of adjacent segments is too large, could cause memory allocation failure" << endl;
        return;
    }
    max_adj_ = max_adj;
}

cv::Mat MeanShift::gaussianKernel(int size) {
    if (size%2 == 0)
        size+= 1;

    cv::Mat1i x_idx, y_idx;
    int half_size = (size-1)/2;
    if (half_size == 0)
        return cv::Mat(1,1, CV_32F, cv::Scalar(1));
    utils::meshGrid(cv::Range(-half_size, half_size), cv::Range(-half_size, half_size), x_idx, y_idx);

    x_idx = x_idx.mul(x_idx);
    y_idx = y_idx.mul(y_idx);
    cv::Mat fx_idx, fy_idx;
    x_idx.convertTo(fx_idx, CV_32F);
    y_idx.convertTo(fy_idx, CV_32F);
    int size2 = size*size;
    cv::Mat kernel;
    kernel = (fx_idx+fy_idx) / (2.0*(float)size2);
    cv::exp(-kernel, kernel);
    kernel = kernel/cv::sum(kernel);

    return kernel;
}


