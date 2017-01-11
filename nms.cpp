#ifndef NMS_HPP
#define NMS_HPP

#include "nms.h"
#include "eigenplus.h"
using namespace std;
using namespace Eigen;

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

    while (I.size() > 0){
        //xx1 = max(x1(i), x1(I(1:last-1)));
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
        vector<double> o_list(o.data(), o.data() + o.rows()*o.cols());
        npwhere_vec(I, o_list, threshold);
    }
}

#endif // NMS_HPP
