//
// Created by sam on 2020-05-03.
//

#ifndef SEGMENTATION_IO_IMAGE_H
#define SEGMENTATION_IO_IMAGE_H

#include <iostream>
#include <experimental/filesystem>
#include "image.h"

namespace io {
    using namespace std;
    using namespace io;
    namespace fs = experimental::filesystem;

    enum Status{SUCCESS, FAIL_DIR, FAIL_EXT, FAIL_FILE};

    bool pathExists(const fs::path &p, fs::file_status s = fs::file_status{}) {
        return fs::status_known(s) ? fs::exists(s) : fs::exists(p);
    }

    bool checkSuffix(const string &suffix, const vector <string> &suffixes) {
        auto iter = find(suffixes.begin(), suffixes.end(), suffix);
        return (iter != suffixes.end() || suffixes.empty());
    }

    int getDataFiles(const string &dir, vector<string> &file_paths, const vector<string> &suffixes = {}) {
        fs::path path(dir);
        string file_name = path.filename().string();

        if (!pathExists(path)) {
            if (!file_paths.empty())
                file_paths.clear();

            file_paths.push_back(dir);
            if (path.extension() == "")
                return FAIL_DIR;
            else
                return FAIL_FILE;
        }

        // passed in dir is a dir to folder
        if (path.extension() == "") {
            string file_dir = path.string();
            if (file_name == ".")
                file_dir = path.parent_path().string();
            for (auto iter = fs::directory_iterator(file_dir); iter != fs::directory_iterator(); ++iter) {
                if (!pathExists(*iter, iter->status()))
                    continue;
                fs::path tmp_path(*iter);
                string tmp_suffix = tmp_path.extension().string();
                if (checkSuffix(tmp_suffix, suffixes))
                    file_paths.push_back(tmp_path.string());
            }
        }
        // passed in dir is a dir to file
        else {
            if (!file_paths.empty())
                file_paths.clear();

            string tmp_suffix = path.extension().string();
            if (checkSuffix(tmp_suffix, suffixes))
                file_paths.push_back(dir);
        }

        if (file_paths.empty())
            return FAIL_EXT;

        return SUCCESS;
    }

    void loadMultiImages(const string &dir, Images &images, const int flag, const vector <string> &suffixes = {}) {
        vector <string> file_paths;
        int status = getDataFiles(dir, file_paths, suffixes);
        if (status == SUCCESS) {
            for (size_t i = 0; i < file_paths.size(); ++i) {
                fs::path path(file_paths[i]);
                cv::Mat cv_img = cv::imread(file_paths[i], flag);
                std::string name = path.filename().string();
                std::string ext = path.extension().string();
                name.erase(name.end()-ext.size(), name.end());
                if (!cv_img.data) {
                    cout << "Could not load image " << name << ext << endl;
                    continue;
                }
                images.addImage(Image(cv_img, name, ext));
                cout << "Loaded image " << name << ext << endl;
            }
        }
        else if (status == FAIL_EXT) {
            cerr << "Could not find file with the specific suffixes" << endl;
        }
        else if (status == FAIL_FILE){
            fs::path path(file_paths[0]);
            std::string file_name = path.filename().string();
            cerr << "The image " << file_name <<  " doesn't exist" << endl;
        }
        else if (status == FAIL_DIR){
            cerr << "The directory " << file_paths[0] << " doesn't exist" << endl;
        }
    }

    Image loadImage(const string &dir, const int flag) {
        fs::path path(dir);
        Image image;

        if (path.extension() == "") {
            cerr << "Input directory is to a folder, maybe you want to use loadMultiImages to load multiple images from a folder" << endl;
        }
        else {
            Images images("temp");
            vector<string> suffixes;
            suffixes.push_back(path.extension().string());
            loadMultiImages(dir, images, flag, suffixes);
            if (!images.empty()) {
                image = images.at(0);
            }
            else {
                cerr << "Could not load image " << path.filename().string() << endl;
            }
        }

        return image;
    }
}

#endif //SEGMENTATION_IO_IMAGE_H
