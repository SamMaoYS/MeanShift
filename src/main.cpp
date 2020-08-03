#include "../include/io_image.h"
#include "opencv2/imgproc.hpp"

int main(int argc, char** argv) {
    Images images("images_0");
    io::loadMultiImages("../data/", images, -1);
    Image a = images.at(0);
    a.show();
}