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
#include "headlayer.h"
using namespace cv;

//#include <iostream>

#ifndef IN_TEST
int main(int args, char* argv[])
{
    if(args  == 0){
       std::cout << "Usage: ./mtcnn.exe [model path]" << std::endl;
       return -1;
    }
    std::cout << argv[1] << std::endl;
    FaceDetector detector;
    detector.initialize(argv[1]);

    ifstream imgfile;
    string imgpath;
    //imgfile.open("/home/duino/project/iactive/mtcnn/mtcnn/imglist.txt");
    imgfile.open("H:/project/qt/mtcnn_cpp/mtcnn/imglist_win.txt");
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
