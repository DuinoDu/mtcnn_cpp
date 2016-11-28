#ifndef NMS_HPP
#define NMS_HPP

#include "eigenplus.h"
using namespace std;
using namespace Eigen;

void nms(MatrixXd &boundingbox, float threshold, string type, vector<int>& pick);

#endif // NMS_HPP
