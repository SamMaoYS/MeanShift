//
// Created by sam on 2020-05-03.
//

#ifndef SEGMENTATION_IO_IMAGE_H
#define SEGMENTATION_IO_IMAGE_H

#include <iostream>
#include "image.h"

namespace io {
    enum Status{SUCCESS, FAIL_DIR, FAIL_EXT, FAIL_FILE};

    void loadMultiImages(const std::string &dir, Images &images, int flag, const std::vector <std::string> &suffixes = {});
    Image loadImage(const std::string &dir, int flag);

    void saveMultiImages(const Images &imgs, const std::string &path);
    void saveImage(const Image &img, const std::string &path);
}

#endif //SEGMENTATION_IO_IMAGE_H
