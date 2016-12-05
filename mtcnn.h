#ifndef _MTCNN_HPP
#define _MTCNN_HPP

#define CPU_ONLY
#include <Eigen/Dense>
#include <igl/sort.h>
#include <igl/find.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <caffe/caffe.hpp>

#include <iostream>
#include <vector>
#include <list>
#include <algorithm>
#include <string>
#include <cmath>
#include <cstdlib>

#include "eigenplus.h"
#include "caffeplus.h"
#include "nms.h"

using namespace std;
using namespace Eigen;
using namespace cv;

void bbreg(MatrixXd &boundingbox, MatrixXd &reg);

void pad(MatrixXd &boundingbox, double w, double h, MatrixXd &result);

void rerec(MatrixXd &boundingbox);

void generateBoundingBox(MatrixXd &map, vector<MatrixXd> &reg, double scale, double threshold, MatrixXd &boxes);

void drawBoxes(Mat &im, MatrixXd &boxes);

void drawBoxes(Mat &im, vector<vector<int>> &boxes);

void _prepareData(shared_ptr<caffe::Net<float>>& net, const Mat& img);

void _prepareData2(shared_ptr<caffe::Net<float>>& net, const vector<Mat>& imgs);

void _stage1(Mat &img_mat, int minsize, shared_ptr<caffe::Net<float>> PNet,
    vector<float> &threshold, bool fastresize, float factor, MatrixXd &total_boxes);

void _stage2(Mat &img_mat, shared_ptr<caffe::Net<float>> RNet, 
    vector<float> &threshold, MatrixXd &total_boxes);

void _stage3(Mat &img_mat, shared_ptr<caffe::Net<float>> ONet, 
    vector<float> &threshold, MatrixXd &total_boxes);

void detect_face(Mat &img_mat, int minsize, 
	shared_ptr<caffe::Net<float>> PNet, shared_ptr<caffe::Net<float>> RNet, shared_ptr<caffe::Net<float>> ONet,
    vector<float> threshold, bool fastresize, float factor, MatrixXd &boxes);

class FaceDetector
{
public:
    FaceDetector(){}
    void initialize(const string& _model_path){ model_path = _model_path; init(); }
    void detect(Mat& _img, vector<vector<int>>& boxes);

private:
    void init();
    string model_path;
    vector<float> threshold;
    float factor;
    int minsize;
    bool fastresize;
    shared_ptr<caffe::Net<float>> PNet, RNet, ONet;
};

#endif
