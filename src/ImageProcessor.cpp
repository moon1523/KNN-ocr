#include "ImageProcessor.h"
#include "Config.h"

#include <vector>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class sortRectByX {
public:
	bool operator()(cv::Rect const& a, cv::Rect const& b) const { return a.x < b.x; }
};

class sortRectByY {
public:
	bool operator()(cv::Rect const& a, cv::Rect const& b) const {return a.y < b.y;	}
};

ImageProcessor::ImageProcessor(const Config& config) :
		_config(config), _debugWindow(false), _debugSkew(false), _debugDigits(false), _debugEdges(false),
		_key(0), powerOn(false), _ocrkVmA(false), _debugPower(false), _debugOCR(false) {
	ofs.open("test1.txt");
}

void ImageProcessor::setInput(cv::Mat& img) { _img = img; }

const std::vector<cv::Mat>& ImageProcessor::getOutput() { return _digits; }
const std::vector<cv::Mat>& ImageProcessor::getOutputkV() { return _digits_kV; }
const std::vector<cv::Mat>& ImageProcessor::getOutputmA() { return _digits_mA; }

void ImageProcessor::debugWindow(bool bval) {
	_debugWindow = bval;
	if(_debugWindow) {
		cv::namedWindow("ImageProcessor");
	}
}
void ImageProcessor::debugSkew(bool bval) { _debugSkew = bval; }
void ImageProcessor::debugEdges(bool bval) { _debugEdges = bval; }
void ImageProcessor::debugDigits(bool bval) { _debugDigits = bval; }
void ImageProcessor::debugPower(bool bval) { _debugPower = bval; }
void ImageProcessor::debugOCR(bool bval) { _debugOCR = bval; }
void ImageProcessor::ocrkVmA(bool bval) { _ocrkVmA = bval; }

int ImageProcessor::showImage() {
	cv::imshow("ImageProcessor", _img);
	_key = cv::waitKey(1);

	return _key;
}

void ImageProcessor::process() {
	_digits.clear();

	// convert to gray
	cvtColor(_img, _imgGray, cv::COLOR_BGR2GRAY);

	// initial rotation to get the digits up
	rotate(_config.getRotationDegrees());

	// detect and correct remaining skew (+- 30 deg)
	if (_debugSkew) {
		float skew_deg = detectSkew();
		rotate(skew_deg);
	}

	// find and isolate counter digits
	findCounterDigits();

	if (_debugWindow) {
		showImage();
	}
}

bool ImageProcessor::process(ROIBox* roi)
{
	_digits.clear();
	_digits_kV.clear();
	_digits_mA.clear();

	// convert to gray
	cvtColor(_img, _imgGray, cv::COLOR_BGR2GRAY);

	// initial rotation to get the digits up
	rotate(_config.getRotationDegrees());

	// detect and correct remaining skew (+- 30 deg)
	if (_debugSkew) {
		float skew_deg = detectSkew();
		rotate(skew_deg);
	}

	// find and isolate counter digits
	findCounterDigits(roi);

	if (_debugWindow) {
		showImage();
	}

	return powerOn;
}

void ImageProcessor::rotate(double rotationDegrees) {
	cv::Mat M = cv::getRotationMatrix2D(cv::Point(_imgGray.cols/2, _imgGray.rows/2), rotationDegrees, 1);
	cv::Mat img_rotated;
	cv::warpAffine(_imgGray, img_rotated, M, _imgGray.size());
	_imgGray = img_rotated;
	if (_debugWindow) {
		cv::warpAffine(_img, img_rotated, M, _img.size());
		_img = img_rotated;
	}
}

void ImageProcessor::drawLines(std::vector<cv::Vec2f>& lines) {
	// draw lines
	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0];
		float theta = lines[i][1];
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		cv::Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
		cv::Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
		cv::line(_img, pt1, pt2, cv::Scalar(255, 0, 0), 1);
	}
}

float ImageProcessor::detectSkew() {
	cv::Mat edges = cannyEdges();

	// find lines
	std::vector<cv::Vec2f> lines;
	cv::HoughLines(edges, lines, 1, CV_PI / 180.f, 140);

	// filter lines by theta and compute average
	std::vector<cv::Vec2f> filteredLines;
	float theta_min = 60.f * CV_PI / 180.f;
	float theta_max = 120.f * CV_PI / 180.0f;
	float theta_avr = 0.f;
	float theta_deg = 0.f;
	for (size_t i=0; i < lines.size(); i++) {
		float theta = lines[i][1];
		if (theta >= theta_min && theta <= theta_max) {
			filteredLines.push_back(lines[i]);
			theta_avr += theta;
		}
	}
	if (filteredLines.size() > 0) {
		theta_avr /= filteredLines.size();
		theta_deg = (theta_avr / CV_PI * 180.f) - 90;
		//printf("detectSkew: %.1f deg", theta_deg);
	} else { std::cout << "failed to detect skew" << std::endl; }

	if (_debugSkew)
		drawLines(filteredLines);

	return theta_deg;
}

cv::Mat ImageProcessor::cannyEdges() {
	cv::Mat edges;
	//detect edges
	cv::Canny(_imgGray, edges, _config.getCannyThreshold1(), _config.getCannyThreshold2());
	return edges;
}

cv::Mat ImageProcessor::binaryFiltering() {
	cv::Mat bin;
	cv::threshold(_imgGray, bin, _config.getBinaryThreshold(), 255, cv::THRESH_BINARY);
	return bin;
}

void ImageProcessor::findAlignedBoxes(std::vector<cv::Rect>::const_iterator begin,
		std::vector<cv::Rect>::const_iterator end, std::vector<cv::Rect>& result) {
	std::vector<cv::Rect>::const_iterator it = begin;

	cv::Rect start = *it;
	++it;
	result.push_back(start);

	for (; it != end; ++it) {
		if (abs(start.y - it->y) < _config.getDigitYAlignment() && abs(start.height - it->height) < 5) {
			result.push_back(*it);
		}
	}
}

void ImageProcessor::filterContours(std::vector<std::vector<cv::Point> >& contours,
        std::vector<cv::Rect>& boundingBoxes, std::vector<std::vector<cv::Point> >& filteredContours) {
	// filter contours by bounding rect size
	for (size_t i=0; i<contours.size(); i++) {
		cv::Rect bounds = cv::boundingRect(contours[i]);
		if (bounds.height > _config.getDigitMinHeight() && bounds.height < _config.getDigitMaxHeight()) {
			boundingBoxes.push_back(bounds);
			filteredContours.push_back(contours[i]);
		}
	}
}

void ImageProcessor::findCounterDigits() {
	// edge image
//	cv::Mat edges = cannyEdges();
	cv::Mat edges = binaryFiltering();
	if (_debugEdges) {
		cv::imshow("edges", edges);
	}
	cv::Mat edges_resize;
	//cv::resize(edges, edges_resize, cv::Size(edges.rows*2, e*resize_factor), 0, 0, INTER_LINEAR);


	cv::Mat img_ret = edges.clone();

	// find contours in whole image
	std::vector<std::vector<cv::Point>> contours, filteredContours;
	std::vector<cv::Rect>boundingBoxes;
	cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // filter contours by bounding rect size
    filterContours(contours, boundingBoxes, filteredContours);

    // find bounding boxes that are aligned at y position
    std::vector<cv::Rect> alignedBoundingBoxes, tmpRes;
    for (std::vector<cv::Rect>::const_iterator ib = boundingBoxes.begin(); ib != boundingBoxes.end(); ++ib) {
        tmpRes.clear();
        findAlignedBoxes(ib, boundingBoxes.end(), tmpRes);
        if (tmpRes.size() > alignedBoundingBoxes.size()) {
            alignedBoundingBoxes = tmpRes;
        }
    }

    // sort bounding boxes from left to right
    std::sort(alignedBoundingBoxes.begin(), alignedBoundingBoxes.end(), sortRectByX());
    // sort bounding boxes from bottom to top
    std::sort(alignedBoundingBoxes.begin(), alignedBoundingBoxes.end(), sortRectByY());

    if (_debugEdges) {
        // draw contours
        cv::Mat cont = cv::Mat::zeros(edges.rows, edges.cols, CV_8UC1);
        cv::drawContours(cont, filteredContours, -1, cv::Scalar(255));
        cv::imshow("contours", cont);
    }

    // cut out found rectangles from edged image
    for (int i = 0; i < alignedBoundingBoxes.size(); ++i) {
        cv::Rect roi = alignedBoundingBoxes[i];
        _digits.push_back(img_ret(roi));
        if (_debugDigits) {
            cv::rectangle(_img, roi, cv::Scalar(0, 255, 0), 2);
        }
    }
}

void ImageProcessor::findCounterDigits(ROIBox* _roi)
{
	cv::Mat edges = binaryFiltering();
	if (_debugEdges) {
		cv::imshow("edges", edges);
	}


	if (_debugPower) {
		cv::Mat powerScreen = edges(_roi->getROIBox()[_roi->getROIBox().size()-1]); // last ROI box must be power screen.

		powerOn = false;
		for (int i=0; i<powerScreen.cols; i++) {
			for (int j=0; j<powerScreen.rows; j++) {
				uchar b = powerScreen.at<cv::Vec3b>(i,j)[0];
				uchar g = powerScreen.at<cv::Vec3b>(i,j)[1];
				uchar r = powerScreen.at<cv::Vec3b>(i,j)[2];
				if ( b > 100 && g > 100 && r > 100 ) { powerOn = true; break;}
			}
			if(powerOn) break;
		}
	}

	cv::Mat img_ret = edges.clone();

	// find contours in whole image
	std::vector<std::vector<cv::Point>> contours, filteredContours;
	std::vector<cv::Rect>boundingBoxes;
	cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	//filter contours by bounding rect size
	filterContours(contours, boundingBoxes, filteredContours);

//	// find bounding boxes that are aligned at y position
//	std::vector<cv::Rect> alignedBoundingBoxes, tmpRes;
//	for (std::vector<cv::Rect>::const_iterator ib = boundingBoxes.begin(); ib != boundingBoxes.end(); ++ib) {
//		tmpRes.clear();
//		findAlignedBoxes(ib, boundingBoxes.end(), tmpRes);
//		if (tmpRes.size() > alignedBoundingBoxes.size()) {
//			alignedBoundingBoxes = tmpRes;
//		}
//	}

	// sort bounding boxes from left to right
	std::sort(boundingBoxes.begin(), boundingBoxes.end(), sortRectByX());


	if (_debugEdges) {
		// draw contours
		cv::Mat cont = cv::Mat::zeros(edges.rows, edges.cols, CV_8UC1);
		cv::drawContours(cont, contours, -1, cv::Scalar(255));
		cv::imshow("contours", cont);
	}


	// cut out found rectangles from edged image
//	std::cout << boundingBoxes.size() << std::endl;
	for (int i = 0; i < boundingBoxes.size(); ++i) {
		cv::Rect roi = boundingBoxes[i];
		_digits.push_back(img_ret(roi));

		if (_debugDigits) {
			cv::rectangle(_img, roi, cv::Scalar(0, 255, 0), 1);
		}
		ofs << _digits[i] << std::endl;
	}


	if (_ocrkVmA) {
		cv::Mat voltScreen  = edges(_roi->getROIBox()[0]);
		cv::Mat currScreen  = edges(_roi->getROIBox()[1]);

		std::vector<std::vector<cv::Point>> kVcontours, kVfilteredContours;
		std::vector<cv::Rect> kVboundingBoxes;
		cv::findContours(voltScreen, kVcontours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		filterContours(kVcontours, kVboundingBoxes, kVfilteredContours);
		std::sort(kVboundingBoxes.begin(), kVboundingBoxes.end(), sortRectByX());

		for (int i=0; i < kVboundingBoxes.size(); ++i) {
			cv::Rect kVroi = kVboundingBoxes[i];
			_digits_kV.push_back(voltScreen(kVroi));
		}

		std::vector<std::vector<cv::Point>> mAcontours, mAfilteredContours;
		std::vector<cv::Rect> mAboundingBoxes;
		cv::findContours(currScreen, mAcontours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		filterContours(mAcontours, mAboundingBoxes, mAfilteredContours);
		std::sort(mAboundingBoxes.begin(), mAboundingBoxes.end(), sortRectByX());

		for (int i=0; i < mAboundingBoxes.size(); ++i) {
			cv::Rect mAroi = mAboundingBoxes[i];
			_digits_mA.push_back(currScreen(mAroi));
		}
	}
}
