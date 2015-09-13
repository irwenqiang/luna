#include "LogisticRegression.h""
#include "matrix.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <boost/algorithm/string.hpp>
using namespace std;
void LogisticRegressionProblem::LoadFeature(const char* featureFile){
    int cur_row = 0;
    ifstream ifs(featureFile);
    string line;
    while(getline(ifs, line)) {

        const char* pos = line.c_str();
        int i = 0;

        dVector feature = dVector(num_feas);

        for(; pos - 1 != NULL && pos[0] != '\0' && pos[0] != '#';
            pos = strchr(pos, ',') + 1) {
            float value = atof(pos);
            feature[i++] = value;
        }

        features[cur_row] = feature;

        cur_row++;
    }
}

void LogisticRegressionProblem::LoadLabel(const char* labelFile) {
    int cur_row = 0;
    ifstream ifs(labelFile);
    double temp = 0;
    while(ifs >> temp) {
        labels[cur_row++] = temp;
    }
}


void LogisticRegressionProblem::LoadInstance(const char* featureFile, const char* labelFile) {
    LoadFeature(featureFile);
    LoadLabel(labelFile);
}

void LogisticRegressionProblem::DumpFeature(const char* featureFile) {
    ofstream ofs(featureFile);
    for(size_t i = 0; i < num_ins; i++) {
        for (size_t j = 0; j < num_feas; j++) {
            ofs << features[i][j] << ",";
        }
        ofs << endl;
    }
}

void LogisticRegressionProblem::DumpLabel(const char* labelFile){
    ofstream ofs(labelFile);//target 0 or 1
    for(size_t i = 0; i < num_ins; i++) {
        ofs << labels[i] << endl;
    }
}

double LogisticRegressionProblem::LearningGD(double alpha, double l2, double l1){
    int max_iters = 4000;
//    memset(&weights, .0, sizeof(double)* num_feas);
    dVector prev_weights = weights;
    dVector predicts = dVector(num_ins);
    double prev_bias = .0;
    double last_mrse = 1e10;

    for(int iter = 0; iter < max_iters; ++iter) {
        double mrse = 0;
        for(int i = 0; i < num_ins; ++i) {
            predicts[i] = predict(features[i]);
            mrse += (labels[i] - predicts[i]) * (labels[i] - predicts[i]);
        }
        if (last_mrse - mrse < 0.0001){
            return mrse;
        }
        last_mrse = mrse;
        std::swap(prev_weights, weights);

        bias = prev_bias;
        //update each weight
        for(int k = 0; k < num_feas; ++k) {
            double gradient = 0.0;
            for(int i = 0; i < num_ins; ++i) {
                gradient += (predicts[i] - labels[i]) * features[i][k];
            }

            if (prev_weights[k] > 0)
                weights[k] = prev_weights[k] - alpha * gradient/num_ins - l2;
            else if (prev_weights[k] < 0) {
                weights[k] = prev_weights[k] - alpha * gradient/num_ins + l2;
            } else {
                weights[k] = prev_weights[k] - alpha * gradient/num_ins;
            }
            //if (_weight_new[k] < 11){ _weight_new[k] = 0; }
        }
        //update bias
        double g = 0.0;
        for(int i = 0; i < num_ins; ++i) {
            g += (predicts[i] - labels[i]);
        }
        prev_bias = bias - alpha * g/num_ins - l2 * bias;
    }

    double sum = 0;
    for(int i = 0; i < num_feas; ++i) {
        double minus = prev_weights[i] - weights[i];
        double r = minus * minus;
        sum += r;
    }
    return sqrt(sum);
}

double LogisticRegressionProblem::logloss(const dVector& feature) {
    return 0.0;
}

double LogisticRegressionProblem::sigmoid(const dVector& feature) {
        double x = inner_prod(feature, weights);
        if (x >= 10){
            return 1.0 / (1.0 + exp(-10));
        }else if (x <= -10){
            return 1.0 / (1.0 + exp(10));
        }
        return 1.0 / (1.0 + exp(-x));
}

double LogisticRegressionProblem::predict(const dVector& feature) {
    return sigmoid(feature) + bias;
}

double LogisticRegressionProblem::Eval(const dVector& input, dVector& gradient) {
    return .0;
}

void LogisticRegressionProblem::SaveModel(std::ostream& os) {
    os << "b:" << bias << " ";
    for(int i = 0; i < num_feas; ++i)
        os <<  i << ":" << weights[i] << " " ;
    os << endl;
}



