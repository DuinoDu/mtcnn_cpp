#define CPU_ONLY
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>

#include "mtcnn.h"
using namespace cv;

#ifndef IN_TEST
int main(int args, char* argv[])
{
    FaceDetector detector;
    detector.initialize(argv[1]);

    ifstream imgfile;
    string imgpath;
    imgfile.open("/home/duino/project/iactive/mtcnn/mtcnn/imglist.txt");
	if (imgfile.is_open()){ 
        while (!imgfile.eof()){
            imgfile >> imgpath;
            cout << imgpath << endl;

            Mat img = imread(imgpath);
            vector<vector<int>> boxes;
            detector.detect(img, boxes);
            drawBoxes(img, boxes);
            imshow("test", img);
            waitKey(0);
        }
    }
    imgfile.close();
    return 0;
}
#else
#include <gtest/gtest.h>
int main(int args, char *argv[])
{
    testing::InitGoogleTest(&args, argv);
    return RUN_ALL_TESTS();
}
#endif
