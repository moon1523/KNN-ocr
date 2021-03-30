#ifndef INCLUDE_IMAGEPROCESSOR_H_
#define INCLUDE_IMAGEPROCESSOR_H_

#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/imgproc/imgproc.hpp>

#include "ImageInput.h"
#include "Config.h"


class ROIBox
{
public:
	ROIBox() : roiBox(0) { }
	std::vector<cv::Rect> getROIBox() { return roiBox; }
	void setROIBox(std::vector<cv::Rect>& _roiBox) { 
		roiBox = _roiBox; std::cout << "ROI box #: " << roiBox.size() << std::endl; 
	}
private:
	std::vector<cv::Rect> roiBox;
};

class ImageProcessor {
public:
	ImageProcessor(const Config& config);

	void setOrientation(int rotationDegrees);
	void setInput(cv::Mat& img);
	void process();
	bool process(ROIBox* roi);
	const std::vector<cv::Mat>& getOutput();
	const std::vector<cv::Mat>& getOutputkV();
	const std::vector<cv::Mat>& getOutputmA();

	void debugWindow(bool bval = true);
	void debugSkew(bool bval = true);
	void debugEdges(bool bval = true);
	void debugDigits(bool bval = true);
	void debugPower(bool bval = true);
	void debugOCR(bool bval = true);
	void ocrkVmA(bool bval = true);
	int  showImage();
	void saveConfig();
	void loadConfig();

	int getKey() { return _key; }
	bool getpowerOn() { return powerOn; }

private:
	void rotate(double rotationDegrees);
	void findCounterDigits();
	void findCounterDigits(ROIBox* roi);
	void findAlignedBoxes(std::vector<cv::Rect>::const_iterator begin,
						  std::vector<cv::Rect>::const_iterator end, std::vector<cv::Rect>& result);
	float detectSkew();
	void drawLines(std::vector<cv::Vec2f>& lines);
	void drawLines(std::vector<cv::Vec4i>& lines, int xoff=0, int yoff=0);
	cv::Mat cannyEdges();
	cv::Mat binaryFiltering();
	void filterContours(std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Rect>& boundingBoxes,
			std::vector<std::vector<cv::Point>>& filteredContours);


	cv::Mat _img;
	cv::Mat _imgGray;
	cv::Mat _imgBin;
	std::vector<cv::Mat> _digits;
	std::vector<cv::Mat> _digits_kV;
	std::vector<cv::Mat> _digits_mA;
	Config _config;
	bool _debugWindow;
	bool _debugSkew;
	bool _debugEdges;
	bool _debugDigits;
	bool _debugPower;
	bool _debugOCR;
	bool _ocrkVmA;
	bool powerOn;

	int _key;

	std::ofstream ofs;
};

#endif /* INCLUDE_IMAGEPROCESSOR_H_ */
