#include "io_image.h"
#include "mean_shift.h"
#include <omp.h>

int main(int argc, char** argv) {
    Images images_rgb("images_rgb");
    io::loadMultiImages("../data/rgb", images_rgb, -1);

    Images images_depth("images_depth");
    io::loadMultiImages("../data/depth", images_depth, -1);
    for (int i = 0; i < images_depth.size(); ++i) {
        Image img = images_rgb.at(i);
        img.resize(0.4);
        img.show(true);

        double time = omp_get_wtime();
        MeanShift mean_shift(img, 8, 6, 2);
        mean_shift.setMergeMaxAdjacent(5);
        mean_shift.segment();
        time = omp_get_wtime() - time;

        std::cout << "Num Segments: " << mean_shift.getNumSegments() << std::endl;
        std::cout << "Mean Shift Time: " << time << std::endl;

        Images results = mean_shift.getResultImages(MeanShift::OUT_FILTER | MeanShift::OUT_SEGM | MeanShift::OUT_RANDOM);
        for (int j = 0; j < results.size(); ++j) {
            results.at(j).show(false, false);
        }

        Images large_areas = mean_shift.getResultImages(MeanShift::OUT_LA);
        Image tmp = images_rgb.at(i);
        tmp.setName(tmp.getName()+"_masked");
        tmp.resize(0.4);
        for (int j = 0; j < large_areas.size(); ++j) {
            tmp.getCVImage().setTo(0, large_areas.at(j).getCVImage()!=0);
        }
        tmp.show(true, true);

//        io::saveMultiImages(results, "../data/result/");
//        win::waitKey(27);
        win::destroyAllWindows();
    }
}