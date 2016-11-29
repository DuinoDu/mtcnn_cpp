#include "eigenplus.h"

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
    MatrixXd out(I.size() - 1, M.cols());
    for (int i = 0; i < I.size() - 1; i++){
        out.row(i) = M.row(I[i]);
    }
    return out;
}

void npwhere_vec(vector<int> &index, const vector<double> &value, const double threshold)
{
    vector<int> out;
    auto i = index.begin();
    auto j = value.begin();
    for (; i != index.end(), j != value.end(); i++, j++){
        if (*j <= threshold){
            out.push_back(*i);
        }
    }
    index.resize(out.size());
    index = out;
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
