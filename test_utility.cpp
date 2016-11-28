#include <gtest/gtest.h>

#include "utility.hpp"
#include <fstream>

void loadFromFile(string filename, MatrixXd& X)
{
    ifstream fin(filename);
    if (fin.is_open()){
        for (int row = 0; row < X.rows(); row++)
            for (int col = 0; col < X.cols(); col++){
                double item = 0.0;
                fin >> item;
                X(row, col) = item;
            }
        fin.close();
    }
}

TEST(bbreg, test_bbreg)
{
    MatrixXd total_boxes;
    total_boxes.resize(2, 5);
    total_boxes << 205, 73, 400, 268, 0.98384535,
    	209, 101, 367, 259, 0.97880608;
    MatrixXd mv;
    mv.resize(4, 2);
    mv << -0.04941138, -0.07724266,
    	0.0447434, -0.08641055,
    	-0.28392452, -0.1872426,
    	0.03337108, 0.05036401;
    MatrixXd mv_t;
    mv_t.resize(2,4);
    mv_t = mv.transpose();
    Matrix<double, 2, 5> boxes_p;
    boxes_p << 195.31536902, 81.76970568, 344.35079408, 274.54073162, 0.98384535,
    	196.71841745, 87.26072219, 337.2284270, 267.00787819, 0.97880608; 

    bbreg(total_boxes, mv_t); // how to use reference here?
    for (int i = 0; i < total_boxes.size(); i++){
    	EXPECT_EQ((int)total_boxes(i), (int)boxes_p(i));
    }
}

TEST(pad, test_pad)
{
    MatrixXd total_boxes;
    total_boxes.resize(3, 5);
    total_boxes <<208, 100, 397, 289, 0.99878901,
    	198, 76, 374, 252, 0.99813914,
    	201, 62, 418, 280, 0.99703938;
    double w = 450;
    double h = 431;
    
    Matrix<double, 3, 1> dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph;
    dy << 0, 0, 0;
    edy << 189., 176., 218.;
    dx << 0, 0, 0;
    edx << 189., 176., 217.;
    y << 99., 75., 61.;
    ey << 288., 251., 279.;
    x << 207., 197., 200.;
    ex << 396., 373., 417.;
    tmpw << 190., 177., 218.;
    tmph << 190., 177., 219.;

    MatrixXd result;
    pad(total_boxes, w, h, result);
    for (int i = 0; i < result.rows(); i++){
    	EXPECT_EQ((int)result(i, 0) , (int)dy(i,0));
    	EXPECT_EQ((int)result(i, 1) , (int)edy(i,0));
    	EXPECT_EQ((int)result(i, 2) , (int)dx(i,0));
    	EXPECT_EQ((int)result(i, 3) , (int)edx(i,0));
    	EXPECT_EQ((int)result(i, 4) , (int)y(i,0));
    	EXPECT_EQ((int)result(i, 5) , (int)ey(i,0));
    	EXPECT_EQ((int)result(i, 6) , (int)x(i,0));
    	EXPECT_EQ((int)result(i, 7) , (int)ex(i,0));
    	EXPECT_EQ((int)result(i, 8) , (int)tmpw(i,0));
    	EXPECT_EQ((int)result(i, 9) , (int)tmph(i,0));
    }
}

TEST(rerec, test_rerec)
{
    MatrixXd total_boxes;
    total_boxes.resize(5, 5);
    total_boxes << 
    	230.07784033, 100.35094793, 375.76201081, 289.35045105, 0.99878901,
    	217.63145834, 76.52293204, 355.90327752, 252.88063219, 0.99813914,
    	219.13960473, 62.8926762, 400.67869663, 280.61021234, 0.99703938,
    	246.17034657, 275.92842653, 311.22883457, 362.11322095, 0.99665654,
    	238.78208129, 189.81249212, 304.58434872, 268.79625906, 0.9940117;
    Matrix<double, 5, 5> out;
    out << 
    	208.42017401, 100.35094793, 397.41967713, 289.35045105, 0.99878901,
    	198.58851785, 76.52293204, 374.94621801, 252.88063219, 0.99813914,
    	201.05038261, 62.8926762, 418.76791875, 280.61021234, 0.99703938,
    	235.60719337, 275.92842653, 321.79198778, 362.11322095, 0.99665654,
    	232.19133154, 189.81249212, 311.17509848, 268.79625906, 0.9940117;
    rerec(total_boxes);
    for (int i = 0; i < total_boxes.size(); i++){
    	EXPECT_EQ((int)total_boxes(i) , (int)out(i));
    }
}

TEST(generateBoxes, test_generateBoxes)
{
    MatrixXd map = MatrixXd::Zero(130, 125);
    string filePath("/home/duino/project/iactive/mtcnn/");
    loadFromFile(filePath+"test_data/map.out", map);
    MatrixXd reg0 = MatrixXd::Zero(130, 125);
    MatrixXd reg1 = MatrixXd::Zero(130, 125);
    MatrixXd reg2 = MatrixXd::Zero(130, 125);
    MatrixXd reg3 = MatrixXd::Zero(130, 125);
    loadFromFile(filePath+"test_data/reg0.out", reg0);
    loadFromFile(filePath+"test_data/reg1.out", reg1);
    loadFromFile(filePath+"test_data/reg2.out", reg2);
    loadFromFile(filePath+"test_data/reg3.out", reg3);
    vector<MatrixXd> reg;
    reg.push_back(reg0);
    reg.push_back(reg1);
    reg.push_back(reg2);
    reg.push_back(reg3);
    MatrixXd boxes = MatrixXd::Zero(32, 9);
    loadFromFile(filePath+"test_data/boxes.out", boxes);

    double scale = 0.6;
    double threshold = 0.6;
    MatrixXd out;
    generateBoundingBox(map, reg, scale, threshold, out);
    EXPECT_EQ(boxes.rows() , out.rows());
    EXPECT_EQ(boxes.cols() , out.cols());

    for (int i = 0; i < boxes.size(); i++){
        EXPECT_EQ(boxes(i) , out(i));
    }
}

TEST(drawBoxes, test_drawBoxes)
{
    string filePath("/home/duino/project/iactive/mtcnn/");
    Mat im = imread(filePath+"test_data/test.jpg");
    MatrixXd boxes;
    boxes.resize(3, 4);
    boxes << 10, 10, 20, 20,
        20, 20, 40, 40,
        60, 60, 100, 100;
    drawBoxes(im, boxes);
    //imshow("drawBoxes test", im);
    //waitKey(0);
}

int main(int args, char *argv[])
{
    testing::InitGoogleTest(&args, argv);
    return RUN_ALL_TESTS();
}
