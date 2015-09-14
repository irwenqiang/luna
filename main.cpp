#include "sgd_lr.cpp"
#include "lr.cpp"
#include "lr_sgd.cpp"
#include "LogisticRegression.h"
#include<iostream>
using namespace std;

int run_sgd(const char* feature, const char* target, int row, int col, const char* test_feature);
int run_lr(const char* feature, const char* target, int row, int col, const char* test_feature);
int run_ilr(const char* feature, const char* target, int row, int col, const char* test_feature);

int main (int argc, const char* argv[]) {

    const char* feature = "E:\\mlcode\\lr\\data\\train.csv";
    const char* featureDump = "E:\\mlcode\\lr\\data\\trainDump.csv";
    const char* target = "E:\\mlcode\\lr\\data\\trainLabels.csv";
    const char* targetDump = "E:\\mlcode\\lr\\data\\trainLabelsDump.csv";
    int row = 1000;
    int col = 40;

    //run_sgd(feature, target, row, col, "E:\\mlcode\\lr\\data\\test.csv");
    //run_sgd(feature, target, row, col, NULL);
    run_lr(feature, target, row, col, NULL);
    //run_ilr(feature, target, row, col, NULL);

}

int run_ilr(const char* feature, const char* target, int row, int col, const char* test_feature) {
    /*
    LogisticRegressionProblem *prob = new LogisticRegressionProblem(row, col);
    prob->LoadFeature(feature);
    //prob->DumpFeature(featureDump);
    prob->LoadLabel(target);
    //prob->DumpLabel(targetDump);
    prob->LearningGD(0.001);
    prob->SaveModel(std::cout);

    double** confuse = dmatrix(2, 2);
    for(int i = 0; i < row; ++i) {
        int pred = prob->predict(prob->features[i]) > 0.5 ? 1 : 0;
        int label = (int)prob->labels[i];
        confuse[label][pred]++;
    }
    cout << "L\t0\t1\tprecision\tsupprt<-prdict\n";
    double label0 = confuse[0][0] + confuse[0][1];
    double label1 = confuse[1][0] + confuse[1][1];
    cout << "0\t" << confuse[0][0] << "\t" << confuse[0][1] << "\t" << confuse[0][0] / label0 << "\t" << label0 << endl;
    cout << "1\t" << confuse[1][0] << "\t" << confuse[1][1] << "\t" << confuse[1][1] / label1 << "\t" << label1 << endl;

    return 0;
    */
    return 0;
}


int run_sgd(const char* feature, const char* target, int row, int col, const char* test_feature) {

    double**x = dmatrix(row, col);
    double* y = dvector(row);
    csv_load_feature(feature, x);
    load_target(target, y);
    SGD_LR model(col);
    //sample x and y
    double alpha = model.backtracking(x, row, col, y);
    model.fit(x, row, col, y, alpha);
    model.save(std::cout);
    double** confuse = dmatrix(2, 2);
    for(int i = 0; i < row; ++i) {
        int pred = model.binary(x[i]);
        int label = (int)y[i];
        confuse[label][pred]++;
    }
    cout << "L\t0\t1\tprecision\tsupprt<-prdict\n";
    double label0 = confuse[0][0] + confuse[0][1];
    double label1 = confuse[1][0] + confuse[1][1];
    cout << "0\t" << confuse[0][0] << "\t" << confuse[0][1] << "\t" << confuse[0][0] / label0 << "\t" << label0 << endl;
    cout << "1\t" << confuse[1][0] << "\t" << confuse[1][1] << "\t" << confuse[1][1] / label1 << "\t" << label1 << endl;
    if(NULL != test_feature) {
        ifstream test(test_feature);
        string line;
        while(getline(test, line)) {
            csv_read(line.c_str(), x[0]);
            cerr << model.binary(x[0]) << endl;
        }
    }
    free_matrix(x, row);
    free_vector(y);
    return 0;
}

int run_lr(const char* feature, const char* target, int row, int col, const char* test_feature) {
    double**x = dmatrix(row, col);
    double* y = dvector(row);

    csv_load_feature(feature, x);
    csv_dump_feature("E:\\mlcode\\lr\\data\\train_dump.csv", x, row, col);
    load_target(target, y);
    dump_target("E:\\mlcode\\lr\\data\\target_dump.csv", y, row);
    LR model(col);
    model.fit(x, row, col, y, 0.1);
    model.save(std::cout);
    double** confuse = dmatrix(2, 2);
    for(int i = 0; i < row; ++i) {
        int pred = model.binary(x[i]);
        int label = (int)y[i];
        confuse[label][pred]++;
    }
    cout << "L\t0\t1\tprecision\tsupprt<-prdict\n";
    double label0 = confuse[0][0] + confuse[0][1];
    double label1 = confuse[1][0] + confuse[1][1];
    cout << "0\t" << confuse[0][0] << "\t" << confuse[0][1] << "\t" << confuse[0][0]/label0 << "\t" << label0 << endl;
    cout << "1\t" << confuse[1][0] << "\t" << confuse[1][1] << "\t" << confuse[1][1]/label1 << "\t" << label1 << endl;
    if(NULL != test_feature) {
        ifstream test(test_feature);
        string line;
        while(getline(test, line)) {
            csv_read(line.c_str(), x[0]);
            cerr << model.binary(x[0]) << endl;
        }
    }
    free_matrix(x, row);
    free_vector(y);
    return 0;
}

