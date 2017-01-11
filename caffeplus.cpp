#include "caffeplus.h"
#include "eigenplus.h"

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

#ifdef DEBUG_MTCNN
    debug_blob(prob);
#endif

    assert(prob->channels() == 2);
    int num = prob->num();

    // convert to score
    float* data = prob->mutable_cpu_data();
    data++;
    for (int i = 0; i < num; i++){
        cout << *data << endl;
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
    int height = blob->height();
    int width = blob->width();
    cout << "\n\nnum:" << num << endl;
    cout << "channels:" << channels << endl;
    cout << "height:" << height << endl;
    cout << "width:" << width << endl;

    float* data = blob->mutable_cpu_data();
    for (int i = 0; i < std::min(num, 3); i++){
        cout << "##########" << endl;
        cout << "# num " << i << "  #";
        cout << "\n##########" << endl;
        for (int j = 0; j < std::min(channels, 3); j++){
            cout << "*****************channels " << j << " *****************" << endl;
            for (int k = 0; k < std::min(width, 3); k++){
                for (int m = 0; m < std::min(height, 3); m++){
                    cout << *(data + m + k*width + j*width*height + i*channels*width*height) << " ";
                }
                cout << endl;
            }
        }
    }
}

void printMatrix(const MatrixXd &M, const string &name)
{
    cout << name << endl << "size: " << M.rows() << "*" << M.cols() << endl;
    cout << M << endl;
}

