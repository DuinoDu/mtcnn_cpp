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

#include "utility.hpp"
using namespace cv;

#ifndef IN_TEST
int main(int args, char* argv[])
{
	//string imglistfile = "H:/project/vs/caffe/windows/mtcnn/mtcnn/imglist.txt";
	//string imglistfile = "H:/project/vs/caffe/windows/mtcnn/mtcnn/file.txt";
	string imglistfile(argv[1]);

	int minsize = 20;
	string caffe_model_path = "H:/project/vs/caffe/windows/mtcnn/mtcnn/model";
	vector<float> threshold;
	threshold.push_back(0.6);
	threshold.push_back(0.7);
	threshold.push_back(0.7);
	float factor = 0.709;

#ifdef CPU_ONLY
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif
	
	shared_ptr<caffe::Net<float>> PNet;
	shared_ptr<caffe::Net<float>> RNet;
	shared_ptr<caffe::Net<float>> ONet;
	
	PNet.reset(new caffe::Net<float>(caffe_model_path + "/det1.prototxt", caffe::TEST));
	PNet->CopyTrainedLayersFrom(caffe_model_path + "/det1.caffemodel");
	RNet.reset(new caffe::Net<float>(caffe_model_path + "/det2.prototxt", caffe::TEST));
	RNet->CopyTrainedLayersFrom(caffe_model_path + "/det2.caffemodel");
	ONet.reset(new caffe::Net<float>(caffe_model_path + "/det3.prototxt", caffe::TEST));
	ONet->CopyTrainedLayersFrom(caffe_model_path + "/det3.caffemodel");

	ifstream imgfile;
	string imgpath;
	imgfile.open(imglistfile);
	if (imgfile.is_open()){ 
		while (!imgfile.eof()){
			imgfile >> imgpath;
			cout << imgpath << endl;

			Mat img = imread(imgpath);
			cvtColor(img, img, CV_BGR2RGB);
	
			MatrixXd boundingboxes;
			detect_face(img, minsize, PNet, RNet, ONet, threshold, false, factor, boundingboxes);
			drawBoxes(img, boundingboxes);
	
			cvtColor(img, img, CV_RGB2BGR);
			imshow("test", img);
			waitKey(0);
		}
	}
	imgfile.close();
}
#else
#include <gtest/gtest.h>
int main(int args, char *argv[])
{
    testing::InitGoogleTest(&args, argv);
    return RUN_ALL_TESTS();
}
#endif
