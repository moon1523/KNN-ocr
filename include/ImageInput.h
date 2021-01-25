#ifndef INCLUDE_IMAGEINPUT_H_
#define INCLUDE_IMAGEINPUT_H_

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Directory.h"

class ImageInput {
public:
	virtual ~ImageInput();

	virtual bool nextImage() = 0;

	virtual cv::Mat& getImage();
	virtual void setImage(cv::Mat& img);
	virtual time_t getTime();
	virtual void setOutputDir(const std::string& outDir);
	virtual void saveImage();

protected:
	cv::Mat _img;
	time_t _time;
	std::string _outDir;
};
/////

class DirectoryInput: public ImageInput {
public:
	DirectoryInput(const Directory& directory);

	virtual bool nextImage();

private:
	Directory _directory;
	std::list<std::string>::const_iterator _itFilename;
	std::list<std::string>                 _filenameList;
};
////

class CameraInput: public ImageInput {
public:
	CameraInput(int device);

	virtual bool nextImage();

private:
	cv::VideoCapture _capture;
};

#endif
