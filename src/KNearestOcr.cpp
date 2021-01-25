#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>

#include <exception>


#include "KNearestOcr.h"

KNearestOcr::KNearestOcr(const Config& config) :
_pModel(), _config(config) {

}

KNearestOcr::~KNearestOcr() {
}


// Learn a single digit.
int KNearestOcr::learn(const cv::Mat& img) {
	cv::imshow("Learn", img);
	int key = cv::waitKey(0) & 255;
	if (key >= 176 && key <= 185) {
		key -= 128; // numeric keypad
	}
	if ((key >= '0' && key <= '9') || key == '.') {
		std::cout << (float) key - '0' << std::endl;
		std::cout << key << std::endl;
		_responses.push_back(cv::Mat(1, 1, CV_32F, (float) key - '0'));
		_samples.push_back(prepareSample(img));
//		std::cout << char(key) << std::flush;
	}
	return key;
}

// Learn a vector of digits.
int KNearestOcr::learn(const std::vector<cv::Mat>& images) {
	int key = 0;
	for (std::vector<cv::Mat>::const_iterator it = images.begin();
			it < images.end() && key != 's' && key != 'q'; ++it) {
		key = learn(*it);
	}
	return key;
}

bool KNearestOcr::hasTrainingData() {
	return !_samples.empty() && !_responses.empty();
}

// Save training data to file.
void KNearestOcr::saveTrainingData() {
	cv::FileStorage fs(_config.getTrainingDataFilename(), cv::FileStorage::WRITE);
	fs << "samples" << _samples;
	fs << "responses" << _responses;
	fs.release();
}

// Load training data from file and init model.
bool KNearestOcr::loadTrainingData() {
	cv::FileStorage fs(_config.getTrainingDataFilename(), cv::FileStorage::READ);
	if (fs.isOpened()) {
		fs["samples"] >> _samples;
		fs["responses"] >> _responses;
		fs.release();

		initModel();
	} else return false;
	return true;
}

// Recognize a single digit.
char KNearestOcr::recognize(const cv::Mat& img) {
	char cres = '?';
	try {
		if (_pModel.empty()) {
			throw std::runtime_error("Model is not initialized");
		}
		cv::Mat results, neighborResponses, dists;

		float result = _pModel->findNearest(prepareSample(img), 5, results, neighborResponses, dists);

		if (0 == int(neighborResponses.at<float>(0, 0) - neighborResponses.at<float>(0, 1))
				&& dists.at<float>(0, 0) < _config.getOcrMaxDist()) {
			// valid character if both neighbors have the same value and distance is below ocrMaxDist
			cres = '0' + (int) result;
		}
		else {
			std::cout << "OCR rejected: " << (int) result << std::endl;
		}
		std::cout << "results: " << results << std::endl;
		std::cout << "neighborResponses: " << neighborResponses << std::endl;
		std::cout << "dists: " << dists << std::endl;
	}
	catch (std::exception & e) {
		std::cerr << e.what() << std::endl;
	}
	return cres;
}

// Recognize a vector of digits.
std::string KNearestOcr::recognize(const std::vector<cv::Mat>& images) {
	std::string result;
	for (std::vector<cv::Mat>::const_iterator it = images.begin();
			it != images.end(); ++it) {
		result += recognize(*it);
	}
	return result;
}


// Prepare an image of a digit to work as a sample for the model.
cv::Mat KNearestOcr::prepareSample(const cv::Mat& img) {
	cv::Mat roi, sample;
	cv::resize(img, roi, cv::Size(10, 10));
	roi.reshape(1,1).convertTo(sample, CV_32F);
	return sample;
}

// Initialize the model.
void KNearestOcr::initModel() {
	_pModel = cv::ml::KNearest::create();
	// load persistent model
	cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(_samples, cv::ml::ROW_SAMPLE, _responses);
	_pModel->train(trainData);
}

