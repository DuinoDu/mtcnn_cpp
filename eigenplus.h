#ifndef EIGENPLUS_H
#define EIGENPLUS_H

#include <Eigen/Dense>
#include <igl/sort.h>
#include <igl/find.h>

//#include <iostream>
#include <vector>
//#include <algorithm>
//#include <cmath>
//#include <cstdlib>

using namespace std;
using namespace Eigen;

VectorXi _find(MatrixXd A, MatrixXd B);
VectorXi _find(MatrixXd A, double b);
void _find(vector<double>& A, double b, vector<int>& C);

void _fix(MatrixXd &M);

MatrixXd subOneRow(MatrixXd M, int index);

MatrixXd subOneRowRerange(MatrixXd &M, vector<int> &I);

void npwhere_vec(vector<int> &index, const vector<double> &value, const double threshold);

void _select(MatrixXd &src, MatrixXd &dst, const vector<int> &pick);


#endif // EIGENPLUS_H
