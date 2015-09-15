#include <iostream>
#include <vector>

typedef std::vector<double> dVector;
typedef std::vector<std::vector<double> > dMatrix;

class LogisticRegressionProblem {
	public:
		dMatrix features;
		dVector labels;
		dVector weights;
		double bias;

		size_t num_ins;
		size_t num_feas;

	public:
		LogisticRegressionProblem(int n, int d): num_ins(n), num_feas(d) {
			features = dMatrix(n);
			for(int i = 0; i < d; i++) {
				features.push_back(dVector(d));
			}
			labels = dVector(n);
			weights = dVector(d);
			bias = .0;
		}

		void LoadFeature(const char* featureFile);

		void LoadLabel(const char* labelFile);

		void LoadInstance(const char* featureFile, const char* labelFile);

		void DumpFeature(const char* featureFile);

		void DumpLabel(const char* labelFile); 

		double LearningGD(double alpha = 0.01, double l2 = .0, double l1 = .0);

		double LearningSGD(double alpha = 0.01, double l2 = .0, double l1 = .0);

		double Sigmoid(const double x);
		double Logloss(const double p, const double y);
		double Predict(const dVector& feature);
		double Eval();

		void SaveModel(std::ostream& os);
};
