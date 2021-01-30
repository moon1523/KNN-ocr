#ifndef INCLUDE_KNEARESTOCR_H_
#define INCLUDE_KNEARESTOCR_H_

#include <vector>
#include <list>
#include <string>
#include "opencv2/core/version.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include "Config.h"

class KNearestOcr {
public:
	KNearestOcr(const Config& config);
	virtual ~KNearestOcr();

	int learn(const cv::Mat& img);
	int learn(const std::vector<cv::Mat>& images);
	bool hasTrainingData();
	void saveTrainingData();
	bool loadTrainingData();

	char recognize(const cv::Mat& img);
	std::string recognize(const std::vector<cv::Mat>& images);

private:
	cv::Mat prepareSample(const cv::Mat& img);
	void initModel();

	cv::Mat _samples;
	cv::Mat _responses;
	cv::Ptr<cv::ml::KNearest> _pModel;
	Config _config;

	std::ofstream ofs;
};

#endif /* INCLUDE_KNEARESTOCR_H_ */
