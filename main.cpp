#include "LogisticRegression.h"
#include<iostream>
using namespace std;

int run_ilr(const char* feature, const char* target, int row, int col, const char* test_feature);

int main (int argc, const char* argv[]) {

	const char* feature = "/Users/chenwenqiang/coderepo/luna/data/train.csv";
	const char* featureDump = "E:\\mlcode\\lr\\data\\trainDump.csv";
	const char* target = "/Users/chenwenqiang/coderepo/luna/data/trainLabels.csv";
	const char* targetDump = "E:\\mlcode\\lr\\data\\trainLabelsDump.csv";
	int row = 1000;
	int col = 40;

	run_ilr(feature, target, row, col, NULL);
}

int run_ilr(const char* feature, const char* target, int row, int col, const char* test_feature) {
	LogisticRegressionProblem *prob = new LogisticRegressionProblem(row, col);
	prob->LoadFeature(feature);
	//prob->DumpFeature(featureDump);
	prob->LoadLabel(target);
	//prob->DumpLabel(targetDump);
	prob->LearningGD(0.001);
	cout << "foo" << endl;
	prob->SaveModel(std::cout);

	double** confuse = new double*[2];
	for (int i = 0; i < 2; i++) {
		confuse[i] = new double[2];
	}
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
}


