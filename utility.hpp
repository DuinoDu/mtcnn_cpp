#ifndef _UTILITY_HPP
#define _UTILITY_HPP

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

void bbreg(MatrixXd &boundingbox, MatrixXd &reg)
{
	assert(boundingbox.cols() == 5);
	assert(reg.cols() == 4);
    assert(boundingbox.rows() == reg.rows());

    cout << "bb.rows:" << boundingbox.rows() << endl;
    cout << "reg.rows:" << reg.rows() << endl;

    if (reg.rows() == 1){
		cout << "reg.rows == 1" << endl;
	}
	int numOfBB = boundingbox.rows();
    Matrix<double, Dynamic, 1> w = boundingbox.col(2).cast<double>() - boundingbox.col(0).cast<double>() + MatrixXd::Ones(numOfBB, 1);
	Matrix<double, Dynamic, 1> h = boundingbox.col(3).cast<double>() - boundingbox.col(1).cast<double>() + MatrixXd::Ones(numOfBB, 1);
	boundingbox.col(0) += w.cwiseProduct(reg.col(0));
	boundingbox.col(1) += h.cwiseProduct(reg.col(1));
	boundingbox.col(2) += w.cwiseProduct(reg.col(2));
	boundingbox.col(3) += h.cwiseProduct(reg.col(3));
}

void pad(MatrixXd &boundingbox, double w, double h, MatrixXd &result)
{
	assert(boundingbox.cols() == 5);

	int numOfBB = boundingbox.rows();
	result.resize(numOfBB, 10);

	Matrix<double, Dynamic, 1> tmpw = boundingbox.col(2).cast<double>() - boundingbox.col(0).cast<double>() + MatrixXd::Ones(numOfBB, 1);
	Matrix<double, Dynamic, 1> tmph = boundingbox.col(3).cast<double>() - boundingbox.col(1).cast<double>() + MatrixXd::Ones(numOfBB, 1);
	MatrixXd dx = MatrixXd::Ones(numOfBB, 1);
	MatrixXd dy = MatrixXd::Ones(numOfBB, 1);
	Matrix<double, Dynamic, 1> edx = tmpw.replicate(1, 1);
	Matrix<double, Dynamic, 1> edy = tmph.replicate(1, 1);

	auto x = MatrixXd(boundingbox.col(0));
	auto y = MatrixXd(boundingbox.col(1));
	auto ex = MatrixXd(boundingbox.col(2));
	auto ey = MatrixXd(boundingbox.col(3));

	MatrixXd w_matrix;
	w_matrix.resize(ex.rows(), ex.cols());
	w_matrix.fill(w);
	VectorXi tmp = _find(ex, w_matrix);

	for (int i = 0; i < tmp.size(); i++){
		int j = tmp(i);
		edx(j) = -ex(j) + w - 1 + tmpw(j);
		ex(j) = w - 1;
	}

	MatrixXd h_matrix;
	h_matrix.resize(ey.rows(), ey.cols());
	h_matrix.fill(h);
	tmp = _find(ey, h_matrix);
	for (int i = 0; i < tmp.size(); i++){
		int j = tmp(i);
		edy(j) = -ey(j) + h - 1 + tmph(j);
		ey(j) = h - 1;
	}

	MatrixXd one_matrix = MatrixXd::Ones(x.rows(), x.cols());
	tmp = _find(one_matrix, x);
	for (int i = 0; i < tmp.size(); i++){
		int j = tmp(i);
		dx(j) = 2 - x(j);
		x(j) = 1;
	}
	
	tmp = _find(one_matrix, y);
	for (int i = 0; i < tmp.size(); i++){
		int j = tmp(i);
		dy(j) = 2 - y(j);
		y(j) = 1;
	}
	dy -= MatrixXd::Ones(dy.rows(), dy.cols());
	edy -= MatrixXd::Ones(dy.rows(), dy.cols());
	dx -= MatrixXd::Ones(dy.rows(), dy.cols());
	edx -= MatrixXd::Ones(dy.rows(), dy.cols());
	y -= MatrixXd::Ones(dy.rows(), dy.cols());
	ey -= MatrixXd::Ones(dy.rows(), dy.cols());
	x -= MatrixXd::Ones(dy.rows(), dy.cols());
	ex -= MatrixXd::Ones(dy.rows(), dy.cols());
	
	result << dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph;
}

void rerec(MatrixXd &boundingbox)
{
	assert(boundingbox.cols() == 5);
	
	auto w = MatrixXd(boundingbox.col(2) - boundingbox.col(0));
	auto h = MatrixXd(boundingbox.col(3) - boundingbox.col(1));
	auto l = w.cwiseMax(h);
	boundingbox.col(0) += w*0.5 - l*0.5;
	boundingbox.col(1) += h*0.5 - l*0.5;
	MatrixXd ll;
	ll.resize(l.rows(), l.cols() * 2);
	ll << l, l;
	boundingbox.middleCols(2, 2) = boundingbox.middleCols(0, 2) + ll;
}

void generateBoundingBox(MatrixXd &map, vector<MatrixXd> &reg, double scale, double threshold, MatrixXd &boxes)
{
	assert(reg.size() == 4);

	int stride = 2;
    int cellsize = 12;
	
	MatrixXd threshold_matrix = MatrixXd(map.rows(), map.cols());
	threshold_matrix.fill(threshold);
	map -= threshold_matrix;
	map = map.cwiseMax(MatrixXd::Zero(map.rows(), map.cols()));
	MatrixXd I, J, V;
	igl::find(map, I, J, V); // I,J is index, V is value. They are all vectors

	// score
	threshold_matrix.resize(V.size(), 1);
	threshold_matrix.fill(threshold);
	MatrixXd score = V + threshold_matrix;

	// reg
	MatrixXd new_reg;
	new_reg.resize(I.size(), 4);
	for (int i = 0; i < 4; i++){ 
		MatrixXd content = MatrixXd::Zero(I.size(), 1);
		for (int num = 0; num < I.size(); num++){
			content(num) = reg[i](I(num), J(num));
		}
		new_reg.middleCols(i,1) = content;
	}
	
	// boundingbox
	MatrixXd boundingbox;
	boundingbox.resize(I.size(), 2);
	boundingbox << I, J;

	MatrixXd cellsize_m = MatrixXd::Zero(boundingbox.rows(), boundingbox.cols());
	cellsize_m.fill(cellsize);

	MatrixXd bb1 = (stride * boundingbox + MatrixXd::Ones(boundingbox.rows(), boundingbox.cols())) / scale;
	MatrixXd bb2 = (stride * boundingbox + cellsize_m) / scale;

	_fix(bb1);
	_fix(bb2);

	assert(bb1.rows() == bb2.rows());
	assert(bb1.rows() == score.rows());
	assert(bb1.rows() == new_reg.rows());
	assert(bb1.cols() == 2);
	assert(bb2.cols() == 2);
	assert(score.cols() == 1);
	assert(new_reg.cols() == 4);

	boxes.resize(bb1.rows(), 9);
	boxes << bb1, bb2, score, new_reg;

	//cout << "score:\n"<< score << endl;
	//cout << "reg:\n" << new_reg << endl;
	//cout << "bb1:\n" << bb1 << endl;
	//cout << "bb2:\n" << bb2 << endl;
}

void drawBoxes(Mat &im, MatrixXd &boxes)
{
	for (int i = 0; i < boxes.rows(); i++){ 
        rectangle(im, Point((int)boxes(i,0), (int)boxes(i,1)), Point((int)boxes(i,2),
            (int)boxes(i,3)), Scalar(0,255,0));
	}
}

void drawBoxes(Mat &im, vector<vector<int>> &boxes)
{
    for (int i = 0; i < boxes.size(); i++){
        rectangle(im, Point(boxes[i][0], boxes[i][1]), Point(boxes[i][2],
            boxes[i][3]), Scalar(0,255,0));
    }
}

void _prepareData(shared_ptr<caffe::Net<float>>& net, const Mat& img)
{
    // 1. reshape data layer
	int height = img.rows;
	int width = img.cols;
	caffe::Blob<float>* input_layer = net->input_blobs()[0];
	input_layer->Reshape(1, 3, height, width);

    // 2. link input data
	std::vector<Mat> input_channels;
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels.push_back(channel);
		input_data += width * height;
	}

	// 3. put img to data layer 
	Mat sample_float;
	img.convertTo(sample_float, CV_32FC3);
	split(sample_float, input_channels);
	CHECK(reinterpret_cast<float*>(input_channels.at(0).data)
	      == net->input_blobs()[0]->cpu_data())
	  << "Input channels are not wrapping the input layer of the network.";
}

void _prepareData2(shared_ptr<caffe::Net<float>>& net, const vector<Mat>& imgs)
{
	assert(imgs.size() > 0);
	// 1. reshape data layer
	int height = imgs[0].rows;
	int width = imgs[0].cols;
	int numbox = imgs.size();
	caffe::Blob<float>* input_layer = net->input_blobs()[0];
	input_layer->Reshape(numbox, 3, height, width);

    // 1.5 transpose imgs
    for (int i = 0; i < numbox; i++){

    }

	// 2. link input data and put into img data
	vector<vector<Mat>> input_all_imgs;
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < numbox; i++){
		vector<Mat> input_channels;
		for (int j = 0; j < input_layer->channels(); j++){
			Mat channel(height, width, CV_32FC1, input_data);
			input_channels.push_back(channel);
			input_data += width * height;
		}
		split(imgs[i], input_channels);
		input_all_imgs.push_back(input_channels);
	}
	CHECK(reinterpret_cast<float*>(input_all_imgs.at(0).at(0).data)
	      == net->input_blobs()[0]->cpu_data())
	  << "Input channels are not wrapping the input layer of the network.";
}

void _stage1(Mat &img_mat, int minsize, shared_ptr<caffe::Net<float>> PNet,
	vector<float> &threshold, bool fastresize, float factor, MatrixXd &total_boxes)
{
	int factor_count = 0;
	int h = img_mat.rows;
	int w = img_mat.cols;
	int minl = std::min(h, w);
	
	float m = 12.0 / minsize;
	minl *= m;

	// create scale pyramid
	vector<float> scales;
	while (minl >= 12){
		scales.push_back(m * std::pow(factor, factor_count));
		minl *= factor;
		factor_count++;
	}

	for (auto scale : scales){
		int hs = (int)std::ceil(h*scale);
		int ws = (int)std::ceil(w*scale);
		Mat im_data;
		img_mat.convertTo(im_data, CV_32FC3);
		if (fastresize){ 
			im_data = (im_data - 127.5) * 0.0078125;
			resize(im_data, im_data, Size(ws, hs));
		}
		else{
			resize(im_data, im_data, Size(ws, hs));
			im_data = (im_data - 127.5) * 0.0078125;
		}

		CHECK_EQ(PNet->num_inputs(), 1) << "Network should have exactly one input.";
		CHECK_EQ(PNet->num_outputs(), 2) << "Network should have exactly two output.";

        Mat im_t = Mat(im_data.cols, im_data.rows, CV_32F);
        transpose(im_data, im_t);
        _prepareData(PNet, im_t);

        //PNet->Forward();
        PNet->ForwardPrefilled();

		caffe::Blob<float>* conv4_2 = PNet->output_blobs()[0]; // 1*4*height*width
		caffe::Blob<float>* prob1 = PNet->output_blobs()[1]; // 1*2*height*width

		//cout << "PNet prob1 height:" << prob1->height() << endl;
		//cout << "PNet conv4-2 height:" << conv4_2->height() << endl;

		// debug prob1
        //debug_blob(prob1);
        //debug_blob(conv4_2);

		MatrixXd map; 
		vector<MatrixXd> reg;
		convertToMatrix(prob1, conv4_2, map, reg);
		MatrixXd boxes;
		generateBoundingBox(map, reg, scale, threshold[0], boxes);

		if (boxes.rows() > 0){
            vector<int> pick;
            nms(boxes, 0.5, "Union", pick);
            if (pick.size() > 0){
                _select(boxes, boxes, pick);
            }
        }

        MatrixXd t(total_boxes.rows() + boxes.rows(), boxes.cols());
		t << total_boxes,
			boxes;
		total_boxes.resize(t.rows(), t.cols());
		total_boxes << t;
	}
}

void _stage2(Mat &img_mat, shared_ptr<caffe::Net<float>> RNet, 
	vector<float> &threshold, MatrixXd &total_boxes)
{
    Mat im_data;
    img_mat.convertTo(im_data, CV_32FC3);

	vector<int> pick;
	nms(total_boxes, 0.7, "Union", pick);
	_select(total_boxes, total_boxes, pick);
	cout << "[2]: " << total_boxes.rows() << endl;

	// using regression, convert n*9 to n*5
	MatrixXd regh = total_boxes.middleCols(3, 1) - total_boxes.middleCols(1, 1);
	MatrixXd regw = total_boxes.middleCols(2, 1) - total_boxes.middleCols(0, 1);
	MatrixXd t1 = total_boxes.middleCols(0, 1) + regw.cwiseProduct(total_boxes.middleCols(5, 1));
	MatrixXd t2 = total_boxes.middleCols(1, 1) + regh.cwiseProduct(total_boxes.middleCols(6, 1));
	MatrixXd t3 = total_boxes.middleCols(2, 1) + regw.cwiseProduct(total_boxes.middleCols(7, 1));
	MatrixXd t4 = total_boxes.middleCols(3, 1) + regh.cwiseProduct(total_boxes.middleCols(8, 1));
	MatrixXd t5 = total_boxes.middleCols(4, 1);
	total_boxes.resize(total_boxes.rows(), 5);
    total_boxes << t1, t2, t3, t4, t5;
	rerec(total_boxes);
	cout << "[4]: " << total_boxes.rows() << endl;
	MatrixXd pad_params;
	pad(total_boxes, img_mat.cols, img_mat.rows, pad_params);
	// pad_params: 0 dy, 1 edy, 2 dx, 3 edx, 4 y, 5 ey, 6 x, 7 ex, 8 tmpw, 9 tmph;

	vector<Mat> imgs;
	for (int i = 0; i < total_boxes.rows(); i++){
		Mat tmp = Mat::zeros(pad_params.col(9)[i], pad_params.col(8)[i], CV_32FC3);
        tmp = im_data(Range(pad_params.col(4)[i], pad_params.col(5)[i] + 1),
			Range(pad_params.col(6)[i], pad_params.col(7)[i] + 1));
		Mat tmp_resize;
		resize(tmp, tmp_resize, Size(24, 24));
		Mat tmp_float;
		tmp_resize.convertTo(tmp_float, CV_32FC3);
		tmp_float = (tmp_float - 127.5) * 0.0078125;
        transpose(tmp_float, tmp_float);
		imgs.push_back(tmp_float);
    }

    _prepareData2(RNet, imgs);

    //debug_blob(RNet->input_blobs()[0]);

    //RNet->Forward();
    RNet->ForwardPrefilled();
    caffe::Blob<float>* conv5_2 = RNet->output_blobs()[0];
	caffe::Blob<float>* prob1 = RNet->output_blobs()[1]; 

    //debug_blob(conv5_2);
    //debug_blob(prob1);

	//use prob1 to filter total_boxes 
    //score = out['prob1'][:,1]
    vector<double> score;
    convertToVector(prob1, score);
    printVector(score, "score");

    vector<int> pass_t;
	_find(score, threshold[1], pass_t);

    filter(total_boxes, pass_t, score);
    printVector(pass_t, "pass_t");

    cout << "[5]:" << total_boxes.rows() << endl;
	
	// use conv5-2 to bbreg
	MatrixXd mv;
	getMV(conv5_2, mv, pass_t);  // 4*N
    if (total_boxes.rows() > 0){
        bbreg(total_boxes, mv);
        vector<int> pick;
        nms(total_boxes, 0.5, "Union", pick);
        if (pick.size() > 0){
			_select(total_boxes, total_boxes, pick);
        }
		cout << "[7]:" << total_boxes.rows() << endl;
		rerec(total_boxes);
		cout << "[8]:" << total_boxes.rows() << endl;
	}
}

void _stage3(Mat &img_mat, shared_ptr<caffe::Net<float>> ONet, 
	vector<float> &threshold, MatrixXd &total_boxes)
{
	MatrixXd pad_params;
	pad(total_boxes, img_mat.cols, img_mat.rows, pad_params);
	// pad_params: 0 dy, 1 edy, 2 dx, 3 edx, 4 y, 5 ey, 6 x, 7 ex, 8 tmpw, 9 tmph;
	//cout << pad_params;
	
	vector<Mat> imgs;
	for (int i = 0; i < total_boxes.rows(); i++){
		Mat tmp = Mat::zeros(pad_params.col(9)[i], pad_params.col(8)[i], CV_32FC3);
		tmp = img_mat(Range(pad_params.col(4)[i], pad_params.col(5)[i] + 1),
			Range(pad_params.col(6)[i], pad_params.col(7)[i] + 1));
		Mat tmp_resize;
		resize(tmp, tmp_resize, Size(48, 48));
		Mat tmp_float;
		tmp_resize.convertTo(tmp_float, CV_32FC3);
		tmp_float = (tmp_float - 127.5) * 0.0078125;
		imgs.push_back(tmp_float);
    }

	_prepareData2(ONet, imgs);
    //ONet->Forward();
    ONet->ForwardPrefilled();
    caffe::Blob<float>* conv6_2 = ONet->output_blobs()[0]; // 4
    caffe::Blob<float>* conv6_3 = ONet->output_blobs()[1]; // 10
    caffe::Blob<float>* prob1 = ONet->output_blobs()[2]; // 2

	//use prob1 to filter total_boxes 
	vector<double> score;

    convertToVector(prob1, score);
	vector<int> pass_t;
	_find(score, threshold[1], pass_t);
	filter(total_boxes, pass_t, score);
	cout << "[9]:" << total_boxes.rows() << endl;
	
	// use conv6-2 to bbreg
	MatrixXd mv;
	getMV(conv6_2, mv, pass_t);  
	if (total_boxes.rows() > 0){ 
        bbreg(total_boxes, mv);
		cout << "[10]:" << total_boxes.rows() << endl;
		vector<int> pick;
        nms(total_boxes, 0.5, "Min", pick);
        if (pick.size() > 0){
			_select(total_boxes, total_boxes, pick);
        }
		cout << "[11]:" << total_boxes.rows() << endl;
	}
}

void detect_face(Mat &img_mat, int minsize, 
	shared_ptr<caffe::Net<float>> PNet, shared_ptr<caffe::Net<float>> RNet, shared_ptr<caffe::Net<float>> ONet,
	vector<float> threshold, bool fastresize, float factor, MatrixXd &boxes)
{
	MatrixXd total_boxes;
	total_boxes.resize(0, 9);
    _stage1(img_mat, minsize, PNet, threshold, fastresize, factor, total_boxes);
	
    if(total_boxes.rows() > 0)
        _stage2(img_mat, RNet, threshold, total_boxes);

    //if (total_boxes.rows() > 0)
    //    _stage3(img_mat, ONet, threshold, total_boxes);

    //cout << "total_boxes num:" << total_boxes.rows() << endl;
    //cout << total_boxes << endl;
	drawBoxes(img_mat, total_boxes);
    boxes = total_boxes;
}

class FaceDetector
{
public:
    FaceDetector(){}
    void initialize(const string& _model_path){ model_path = _model_path; init(); }
    void detect(Mat& _img, vector<vector<int>>& boxes){
        Mat img;
        _img.copyTo(img);
        cvtColor(img, img, CV_BGR2RGB);

        MatrixXd boundingboxes;
        detect_face(img, minsize, PNet, RNet, ONet, threshold, false, factor, boundingboxes);
        //cout << "boundingboxes:\n" << boundingboxes << endl;
        for (int i = 0; i < boundingboxes.rows(); i++){
            vector<int> box;
            box.resize(4);
            assert(boundingboxes.cols() >= 4);
            for (int j = 0; j < 4; j++){
                box[j] = (int)boundingboxes(i, j);
            }
            boxes.push_back(box);
        }
    }

private:
    void init(){
        threshold.push_back(0.6);
        threshold.push_back(0.7);
        threshold.push_back(0.7);
        factor = 0.709;
        minsize = 20;
        fastresize = false;
#ifdef CPU_ONLY
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif
        PNet.reset(new caffe::Net<float>(model_path + "/det1.prototxt", caffe::TEST));
        PNet->CopyTrainedLayersFrom(model_path + "/det1.caffemodel");
        RNet.reset(new caffe::Net<float>(model_path + "/det2.prototxt", caffe::TEST));
        RNet->CopyTrainedLayersFrom(model_path + "/det2.caffemodel");
        ONet.reset(new caffe::Net<float>(model_path + "/det3.prototxt", caffe::TEST));
        ONet->CopyTrainedLayersFrom(model_path + "/det3.caffemodel");
    }

    string model_path;
    vector<float> threshold;
    float factor;
    int minsize;
    bool fastresize;
    shared_ptr<caffe::Net<float>> PNet, RNet, ONet;
};

#endif
