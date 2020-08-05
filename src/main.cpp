#include "io_image.h"
#include "mean_shift.h"

int main(int argc, char** argv) {
    Images images("images_0");
    io::loadMultiImages("../data/", images, -1);

    for (int i = 0; i < images.size(); ++i) {
        Image img = images.at(i);
        img.show(true, false);

        MeanShift mean_shift(img, 8, 16);
        mean_shift.segment();

        std::cout << "Num Segments: " << mean_shift.getNumSegments() << std::endl;

        Image filtered = mean_shift.getSegmentedImage();
        filtered.show(true, false);
        cv::destroyAllWindows();
    }
}