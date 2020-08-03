#include "io_image.h"
#include "mean_shift.h"

int main(int argc, char** argv) {
    Images images("images_0");
    io::loadMultiImages("../data/", images, -1);
    
    Image baboon = images.at(0);
    baboon.show();

    MeanShift mean_shift(baboon, 8, 6);
    mean_shift.filter();
}