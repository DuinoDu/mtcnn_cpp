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

using namespace std;
using namespace Eigen;
using namespace cv;


VectorXi _find(MatrixXd A, MatrixXd B){
	// find index where A > B
	Matrix<bool, Dynamic, Dynamic> C = A.array() > B.array();
	VectorXi I = VectorXi::LinSpaced(C.size(), 0, C.size() - 1);
	I.conservativeResize(std::stable_partition(
		I.data(), I.data() + I.size(), [&C](int i){return C(i); }) - I.data());
	return I;
}

VectorXi _find(MatrixXd A, double b){ 
	MatrixXd B = MatrixXd(A.rows(), A.cols());
	B.fill(b);
	Matrix<bool, Dynamic, Dynamic> C = A.array() > B.array();
	VectorXi I = VectorXi::LinSpaced(C.size(), 0, C.size() - 1);
	I.conservativeResize(std::stable_partition(
		I.data(), I.data() + I.size(), [&C](int i){return C(i); }) - I.data());
	return I;
}

void _find(vector<double>& A, double b, vector<int>& C)
{
	for (int i = 0; i < A.size(); i++){ 
		if (A.at(i) > b){ 
			C.push_back(i);
		}
	}
}

void _fix(MatrixXd &M){
	for (int i = 0; i < M.cols(); i++){ 
		for (int j = 0; j < M.rows(); j++){ 
			
			int temp = (int)M(j, i);
			
			if (temp > M(j, i)) temp--;
			else if (M(j, i) - temp > 0.9) temp++;
			
			M(j, i) = (double)temp;
		}
	}
}

MatrixXd subOneRow(MatrixXd M, int index){
	assert(M.rows() > index);
	MatrixXd out(M.rows() - 1, M.cols());
	for (int i = 0, j = 0; i < M.rows(), j < out.rows(); ){
		if (i != index){
			out.row(j) = M.row(i);
			i++;
			j++;
		}
		else
			i++;
	}
	return out;
}

MatrixXd subOneRowRerange(MatrixXd &M, vector<int> &I)
{
	MatrixXd out(M.rows() - 1, M.cols());
	for (int i = 0; i < I.size() - 1; i++){
		out.row(i) = M.row(I[i]);
	}
	return out;
}

void bbreg(MatrixXd &boundingbox, MatrixXd &reg)
{
	assert(boundingbox.cols() == 5);
	assert(reg.cols() == 4);
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

void nms(MatrixXd &boundingbox, float threshold, string type, vector<int>& pick)
{
    assert(boundingbox.cols() == 5 || boundingbox.cols() == 9);
	if (boundingbox.rows() < 1) return;

	MatrixXd x1 = MatrixXd(boundingbox.col(0));
	MatrixXd y1 = MatrixXd(boundingbox.col(1));
	MatrixXd x2 = MatrixXd(boundingbox.col(2));
	MatrixXd y2 = MatrixXd(boundingbox.col(3));
	MatrixXd s = MatrixXd(boundingbox.col(4));
	MatrixXd one_vector = MatrixXd::Ones(x1.rows(), 1);
	MatrixXd area = (x2 - x1 + one_vector).cwiseProduct(y2 - y1 + one_vector);
	MatrixXd _vals;
	MatrixXi _I;
	igl::sort(s, 1, true, _vals, _I);
	vector<int> I(_I.data(), _I.data() + _I.rows()*_I.cols());
	while (true){
		cout << "I:" << endl;
		for (auto i : I) cout << i << ", ";
		cout << endl;

		MatrixXd x1_powerful = MatrixXd(I.size() - 1, 1);
		x1_powerful.fill(x1(I.back()));
		MatrixXd xx1 = x1_powerful.cwiseMax(subOneRowRerange(x1, I));

		MatrixXd y1_powerful = MatrixXd(I.size() - 1, 1);
		y1_powerful.fill(y1(I.back()));
		MatrixXd yy1 = y1_powerful.cwiseMax(subOneRowRerange(y1, I));

		MatrixXd x2_powerful = MatrixXd(I.size() - 1, 1);
		x2_powerful.fill(x2(I.back()));
		MatrixXd xx2 = x2_powerful.cwiseMin(subOneRowRerange(x2, I));

		MatrixXd y2_powerful = MatrixXd(I.size() - 1, 1);
		y2_powerful.fill(y2(I.back()));
		MatrixXd yy2 = y2_powerful.cwiseMin(subOneRowRerange(y2, I)); 
	
		auto w = MatrixXd::Zero(I.size() - 1, 1).cwiseMax(xx2-xx1+MatrixXd::Ones(I.size()-1,1));
		auto h = MatrixXd::Zero(I.size() - 1, 1).cwiseMax(yy2-yy1+MatrixXd::Ones(I.size()-1,1));
		auto inter = w.cwiseProduct(h);

		MatrixXd o;
		MatrixXd area_powerful = MatrixXd(I.size() - 1, 1);
		area_powerful.fill(area(I.back()));
		if (type == "Min"){
			o = inter.cwiseQuotient(area_powerful.cwiseMin(subOneRowRerange(area, I)));
		}
		else{
			MatrixXd tmp = area_powerful + subOneRowRerange(area, I) - inter;
			o = inter.cwiseQuotient(tmp);
		}
			
		pick.push_back(I.back());
		
		// I = I[np.where( o <= threshold )]
		vector<int> newI;
		for (int i = 0; i < I.size(); i++){ 
			if (o(I[i]) <= threshold){ 
				newI.push_back(I[i]);
			}
		}
		if (newI.size() == 0)
			break;
		else{
			I.resize(newI.size());
			I = newI;
		}
		/*
		vector<double> o_list(o.data(), o.data() + o.rows()*o.cols());
		auto i = I.begin();
		auto j = o_list.begin();
		for (; i != I.end(), j != o_list.end(); i++, j++){
			if (*j > threshold){
				I.erase(i);
			}
		}
		*/
	}
	cout << "pick:\n";
	for (auto i : pick) cout << i << endl;
	cout << endl;
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

void _select(MatrixXd &src, MatrixXd &dst, const vector<int> &pick)
{
    MatrixXd _src = src.replicate(1,1);
    int new_height = pick.size();
    int new_width = src.cols();
    dst.resize(new_height, new_width);
    for(int i=0; i < pick.size(); i++){
        dst.row(i) = _src.row(pick[i]);
    }
}

void drawBoxes(Mat &im, MatrixXd &boxes)
{
	for (int i = 0; i < boxes.rows(); i++){ 
		rectangle(im, Rect((int)boxes(i,0), (int)boxes(i,1), (int)(boxes(i,2)-boxes(i,0)), 
			(int)(boxes(i,3) - boxes(i,1))), Scalar(0,255,0));
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

void convertToMatrix(caffe::Blob<float>* prob, caffe::Blob<float>* conv, MatrixXd &map, vector<MatrixXd> &reg)
{
	int height = prob->height();
	int width = prob->width();
	
	// convert to map
    float* data = prob->mutable_cpu_data() + height * width;
    Mat prob_mat(height, width, CV_32FC1, data);
    cv2eigen(prob_mat, map);

	//for (int i = 0; i < 20; i++) cout << "prob " << i << ": "  << *(data + i) << endl;
		
	// convert to reg
    data = conv->mutable_cpu_data();
    MatrixXd eachReg;
    eachReg.resize(height, width);
    for(int i=0; i < conv->channels(); i++){
        Mat reg_mat(height, width, CV_32FC1, data);
        cv2eigen(reg_mat, eachReg);
        reg.push_back(eachReg);
		//cout << "===\n";
		//for (int j = 0; j < 20; j++) cout << i << " conv " << j << ": "  << *(data + j) << endl;
        data += height * width;
    }
}

void convertToVector(caffe::Blob<float>* prob, vector<double> &score)
{
	assert(prob->channels() == 2);
	int num = prob->num();
	int channels = prob->channels();

	// convert to score
    float* data = prob->mutable_cpu_data();
	for (int i = 0; i < num; i++){ 
		score.push_back(*data);
		data += 2;
	}
}

void filter(MatrixXd &total_boxes, VectorXi &pass_t, MatrixXd &score)
{
	MatrixXd new_boxes;
	new_boxes.resize(pass_t.size(), 5);
	for (int i = 0; i < pass_t.size(); i++){
		MatrixXd tmp;
		tmp.resize(1, 5);
		tmp << total_boxes(pass_t(i), 0), total_boxes(pass_t(i), 1), total_boxes(pass_t(i), 2), total_boxes(pass_t(i), 3), score(pass_t(i));
		new_boxes.row(i) = tmp;
	}
	total_boxes.resize(pass_t.size(), 5);
	total_boxes << new_boxes;
}

void filter(MatrixXd &total_boxes, vector<int> &pass_t, vector<double> &score)
{
	MatrixXd new_boxes;
	new_boxes.resize(pass_t.size(), 5);
	for (int i = 0; i < pass_t.size(); i++){
		MatrixXd tmp;
		tmp.resize(1, 5);
		tmp << total_boxes(pass_t.at(i), 0), total_boxes(pass_t.at(i), 1), total_boxes(pass_t.at(i), 2), total_boxes(pass_t.at(i), 3), score.at(pass_t.at(i));
		new_boxes.row(i) = tmp;
	}
	total_boxes.resize(pass_t.size(), 5);
	total_boxes << new_boxes;
}

void getMV(caffe::Blob<float>* conv, MatrixXd &mv, vector<int> &pass_t)
{
	int num = conv->num();
	int channels = conv->channels();

	// convert to MatrixXd
	MatrixXd conv_m;
    float* data = conv->mutable_cpu_data();
    Mat conv_mat(num, channels, CV_32FC1, data);
    cv2eigen(conv_mat, conv_m);
	_select(conv_m, mv, pass_t);
}


void debug_blob(caffe::Blob<float>* blob){
	int num = blob->num();
	int channels = blob->channels();
	int height = 10; // blob->height();
	int width = 10; // blob->width();

	float* data = blob->mutable_cpu_data();
	for (int i = 0; i < num; i++){
		cout << "\n\n\n#######" << endl;
		cout << "#  " << i << "  #";
		cout << "#######" << endl;
		for (int j = 0; j < channels; j++){
			cout << "*****************channels " << j << " *****************" << endl;
			for (int k = 0; k < height; k++){
				for (int m = 0; m < width; m++){
					cout << *(data + m + k*width + j*width*height + i*channels*width*height) << " ";
				}
				cout << endl;
			}
		}
	}
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

	cout << "scales: ";
	for (auto i : scales) cout << i << " ";
	cout << endl;

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

        PNet->Forward();
		
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

		cout << "generateBB:" << endl;
		cout << boxes.rows() << "*" << boxes.cols() << endl;
		cout << boxes << endl;
        
		if (boxes.rows() > 0){
            vector<int> pick;
            nms(boxes, 0.5, "Union", pick);
            if (pick.size() > 0){
                _select(boxes, boxes, pick);
            }

			cout << "pick" << endl;
			for (auto i : pick) cout << i << " ";
			cout << endl;
        }
		MatrixXd t(total_boxes.rows() + boxes.rows(), boxes.cols());
		t << total_boxes,
			boxes;
		total_boxes.resize(t.rows(), t.cols());
		total_boxes << t;
		
		cout << "total_boxes:\n";
		cout << total_boxes << endl;
		break;
	}
}

void _stage2(Mat &img_mat, shared_ptr<caffe::Net<float>> RNet, 
	vector<float> &threshold, MatrixXd &total_boxes)
{
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
		tmp = img_mat(Range(pad_params.col(4)[i], pad_params.col(5)[i] + 1),
			Range(pad_params.col(6)[i], pad_params.col(7)[i] + 1));
		Mat tmp_resize;
		resize(tmp, tmp_resize, Size(24, 24));
		Mat tmp_float;
		tmp_resize.convertTo(tmp_float, CV_32FC3);
		tmp_float = (tmp_float - 127.5) * 0.0078125;
		imgs.push_back(tmp_float);
	}

	_prepareData2(RNet, imgs);
	RNet->Forward();
	caffe::Blob<float>* conv5_2 = RNet->output_blobs()[0];
	caffe::Blob<float>* prob1 = RNet->output_blobs()[1]; 

	//use prob1 to filter total_boxes 
	vector<double> score;
	convertToVector(prob1, score);
	vector<int> pass_t;
	_find(score, threshold[1], pass_t);
	filter(total_boxes, pass_t, score);
	cout << "[5]:" << total_boxes.rows() << endl;
	
	// use conv5-2 to bbreg
	MatrixXd mv;
	getMV(conv5_2, mv, pass_t);  // 4*N
	if (total_boxes.rows() > 0){ 
        vector<int> pick;
        nms(total_boxes, 0.5, "Union", pick);
        if (pick.size() > 0){
			_select(total_boxes, total_boxes, pick);
        }
		cout << "[6]:" << total_boxes.rows() << endl;
		//bbreg(total_boxes, mv);
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
	ONet->Forward();
	caffe::Blob<float>* conv6_2 = ONet->output_blobs()[0];
	caffe::Blob<float>* prob1 = ONet->output_blobs()[1]; 
	caffe::Blob<float>* conv6_3 = ONet->output_blobs()[2];

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
		//bbreg(total_boxes, mv);
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

	if (total_boxes.rows() > 0)
		_stage3(img_mat, ONet, threshold, total_boxes);

	cout << "total_boxes num:" << total_boxes.rows() << endl;
	drawBoxes(img_mat, total_boxes);
}
