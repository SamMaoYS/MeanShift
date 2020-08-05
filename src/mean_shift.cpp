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
    this->setMinSize(20);
}

MeanShift::MeanShift(const Image &img, float hs, float hr, int min_size) {
    this->image_ = img.copy();
    this->setSpatialBandwidth(hs);
    this->setRangeBandwidth(hr);
    this->setMinSize(min_size);
}

MeanShift::~MeanShift() {

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
    cout << "Filtered image " << filtered_.getName() << filtered_.getExtension() << " was generated" << endl;
}

void MeanShift::segment() {
    utils::processPrint("Mean Shift Segment Starts");

    if (configure() != SUCCESS)
        return;

    if (filtered_.empty()) {
        filterRGB();
        cout << "Filtered image " << filtered_.getName() << filtered_.getExtension() << " was generated" << endl;
    }

    cluster();
    merge();
    genSegmentedImage();
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

    int rad_s2 = (int)(hs_*hs_);
    float rad_r2 = 3*hr_*hr_;

    uchar * l_ptr = luv[0].ptr();
    uchar * u_ptr = luv[1].ptr();
    uchar * v_ptr = luv[2].ptr();

#pragma omp parallel for collapse(2)
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
            while (iter < 10 && shift > 3) {
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
                if (c1 <= std::numeric_limits<float>::min() || c2 <= std::numeric_limits<float>::min()) {
                    iter++;
                    continue;
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

    float rad_r2 = 3*hr_*hr_;

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
    if (segmented_.empty()) {
        cerr << "Segmented image is not generated, please run segment first" << endl;
        return Image();
    }
    return segmented_;
}

void MeanShift::genSegmentedImage() {
    int height = image_.height();
    int width = image_.width();
    cv::Mat segmented = cv::Mat(height, width, CV_8UC3, cv::Scalar::all(0));

#pragma omp parallel for collapse(2)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cv::Point3f luv= modes_range_[labels_.at<int>(i,j)];
            cv::Vec3b luv_8U;
            luv_8U[0] = (uchar)luv.x;
            luv_8U[1] = (uchar)luv.y;
            luv_8U[2] = (uchar)luv.z;
            segmented.at<cv::Vec3b>(i,j) = luv_8U;
        }
    }

    cv::cvtColor(segmented, segmented, cv::COLOR_Luv2BGR);
    if (image_.channels() == 1)
        cv::cvtColor(segmented, segmented, cv::COLOR_BGR2GRAY);
    segmented_.setImage(segmented);
    segmented_.setName(image_.getName() + "_segmented");
    cout << "Segmented image " << segmented_.getName() << segmented_.getExtension() << " was generated" << endl;
    segmented.release();
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
    int adj_pool_size = num_segms_*10;
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
    float rad_r2 = 3*hr_*hr_;

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


