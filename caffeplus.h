#ifndef CAFFEPLUS_H
#define CAFFEPLUS_H

#define CPU_ONLY
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <caffe/caffe.hpp>

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

using namespace std;
using namespace Eigen;
using namespace cv;

void convertToMatrix(caffe::Blob<float>* prob, caffe::Blob<float>* conv, MatrixXd &map, vector<MatrixXd> &reg);

void convertToVector(caffe::Blob<float>* prob, vector<double> &score);

void filter(MatrixXd &total_boxes, VectorXi &pass_t, MatrixXd &score);

void filter(MatrixXd &total_boxes, vector<int> &pass_t, vector<double> &score);

void getMV(caffe::Blob<float>* conv, MatrixXd &mv, vector<int> &pass_t);

void debug_blob(caffe::Blob<float>* blob);

void printMatrix(const MatrixXd &M, const string &name);

#endif // CAFFEPLUS_H
