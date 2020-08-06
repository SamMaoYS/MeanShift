#include "io_image.h"
#include "mean_shift.h"
#include <omp.h>

int main(int argc, char** argv) {
    Images images("images_0");
    io::loadMultiImages("../data/", images, -1);

    for (int i = 0; i < images.size(); ++i) {
        Image img = images.at(i);
        img.show();

        double time = omp_get_wtime();
        MeanShift mean_shift(img, 8, 16, 100);
        mean_shift.segment();
        time = omp_get_wtime() - time;

        std::cout << "Num Segments: " << mean_shift.getNumSegments() << std::endl;
        std::cout << "Mean Shift Time: " << time << std::endl;

        Images results = mean_shift.getResultImages(MeanShift::OUT_FILTER | MeanShift::OUT_SEGM | MeanShift::OUT_RANDOM);
        results.at(0).show(false);
        results.at(1).show(false);
        results.at(2).show(true);

        io::saveMultiImages(results, "../data/result/");
        win::destroyAllWindows();
    }
}