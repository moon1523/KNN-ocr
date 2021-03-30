#ifndef INCLUDE_FUNCTIONS_H_
#define INCLUDE_FUNCTIONS_H_

#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "KNearestOcr.h"

int DELAY = 500;

cv::Rect cropRect(0,0,0,0);
cv::Point P1(0,0), P2(0,0);
std::vector<cv::Rect> blackBox;
std::vector<cv::Rect> blackPoint;
bool clicked=false;
bool calc=false;

void onMouseCropImage(int event, int x, int y, int f, void *param);
ROIBox* setROIBOX(ImageInput* pImageInput);
void recordData(int cam);


static void testOcr(ImageInput* pImageInput, int cam) {
	auto roi = setROIBOX(pImageInput);
	bool debugOCR(false);

	Config config;
    config.loadConfig();
    ImageProcessor proc(config);
    proc.debugWindow();
    proc.debugDigits();
    proc.ocrkVmA();
    proc.debugPower();

    KNearestOcr ocr(config);
    if (! ocr.loadTrainingData()) {
        std::cout << "Failed to load OCR training data\n";
        return;
    }
    std::cout << "OCR training data loaded.\n";
    std::cout << "<q> to quit.\n";

    cv::Mat imgNone = cv::Mat::zeros(pImageInput->getImage().rows, pImageInput->getImage().cols, CV_8UC3);

	// Recording ============
	double fps = 1/(DELAY * 0.001);
	int width = pImageInput->getImage().cols;
	int height = pImageInput->getImage().rows;
	std::cout << width << " " << height << std::endl;
	int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

	cv::VideoWriter outputVideo;
	outputVideo.open("record.avi", fourcc, fps, cv::Size(width, height), true);

	if (!outputVideo.isOpened()) { std::cerr << "Recording Initialization Error" << std::endl; exit(1); }
	// ===============


    int frameNo(0);
    while (1) {
		pImageInput->nextImage();
    	std::cout << "Frame " << frameNo++ << std::endl;
		cv::Mat imgCopy = imgNone;
		
		outputVideo.write(pImageInput->getImage());

		for (int k=0; k<blackBox.size(); k++) {
			for (int i=blackBox[k].x; i< blackBox[k].x + blackBox[k].width; i++) {
				for (int j=blackBox[k].y; j<blackBox[k].y + blackBox[k].height; j++) {
					imgCopy.at<cv::Vec3b>(j,i) = pImageInput->getImage().at<cv::Vec3b>(j,i);
				}
			}
		}
    	proc.setInput(imgCopy);
        proc.process(roi);
        bool powerOn = proc.getpowerOn();

        std::cout << "######### KNN RESULTS ########" << std::endl;
//        std::cout << ">> Voltage OCR -----" << std::endl;
        std::string voltage = ocr.recognize(proc.getOutputkV());
//        std::cout << ">> Current OCR -----" << std::endl;
        std::string current = ocr.recognize(proc.getOutputmA());


        if (voltage.find('.') != std::string::npos || current.find('.') != std::string::npos
		 	||voltage.empty() || current.empty()) {
        	std::cout << "######### OCR RESULTS ########" << std::endl;
			if(powerOn)	std::cout << "Power Status: On" << std::endl;
			else		std::cout << "Power Status: Off" << std::endl;
			std::cout << "Tube voltage (kV) : " << voltage << std::endl;
			std::cout << "Tube current (mA) : " << current << std::endl;
			std::cout << "!!WARNING!! point is recognized or character is not recognized" << std::endl;
        }
        else {
        	float currentF = (float)stoi(current) * 0.1;
			std::cout << "######### OCR RESULTS ########" << std::endl;
			if(powerOn)	std::cout << "Power Status: On" << std::endl;
			else		std::cout << "Power Status: Off" << std::endl;
			std::cout << "Tube voltage (kV) : " << stoi(voltage) << std::endl;
			std::cout << "Tube current (mA) : " << currentF << std::endl << std::endl;
        }

        int key = cv::waitKey(DELAY) & 255;

        if (key == 'q') {
            std::cout << "Quit\n";
            break;
        }
    }
}


static void learnOcr(ImageInput* pImageInput) {
	int key = 0;
	auto roi = setROIBOX(pImageInput);

	Config config;
	config.loadConfig();
	ImageProcessor proc(config);
	proc.debugDigits();
	proc.debugEdges();
	proc.debugWindow();

	KNearestOcr ocr(config);
	ocr.loadTrainingData();
	std::cout << "## Entering OCR training mode! ##\n";
	std::cout << "<0>..<9> to answer digit, <.> to answer decimal point, <space> to ignore digit, <s> to save and quit, <q> to quit without saving.\n";

	cv::Mat imgNone = cv::Mat::zeros(pImageInput->getImage().rows, pImageInput->getImage().cols, CV_8UC3);
	while (pImageInput->nextImage()) {
		cv::Mat imgCopy = imgNone;

		for (int k=0; k<blackBox.size(); k++) {
			for (int i=blackBox[k].x; i< blackBox[k].x + blackBox[k].width; i++) {
				for (int j=blackBox[k].y; j<blackBox[k].y + blackBox[k].height; j++) {
					imgCopy.at<cv::Vec3b>(j,i) = pImageInput->getImage().at<cv::Vec3b>(j,i);
				}
			}
		}
		proc.setInput(imgCopy);
		proc.process(roi);

		key = ocr.learn(proc.getOutput());
		std::cout << "...Learn" << std::endl;

		if (key == 'q' || key == 's') {
			std::cout << "Quit\n";
			break;
		}
	}

	if (key != 'q' && ocr.hasTrainingData()) {
		std::cout << "Saving training data\n";
		ocr.saveTrainingData();
	}
}

static void adjustCamera(ImageInput* pImageInput) {
	bool processImage(true);
	int key(0);

	auto roi = setROIBOX(pImageInput);

	Config config;
	config.loadConfig();
	ImageProcessor proc(config);
	proc.debugDigits();
	proc.debugEdges();
	proc.debugWindow();
	std::cout <<"## ADJUST CAMERA ##\n";
	std::cout << "<r>, <p> to select raw or processed image, <s> to save config and quit, <q> to quit without saving.\n";

	cv::Mat imgNone = cv::Mat::zeros(pImageInput->getImage().rows, pImageInput->getImage().cols, CV_8UC3);
	while (pImageInput->nextImage()) {
		cv::Mat imgCopy = imgNone;

		for (int k=0; k<blackBox.size(); k++) {
			for (int i=blackBox[k].x; i< blackBox[k].x + blackBox[k].width; i++) {
				for (int j=blackBox[k].y; j<blackBox[k].y + blackBox[k].height; j++) {
					imgCopy.at<cv::Vec3b>(j,i) = pImageInput->getImage().at<cv::Vec3b>(j,i);
				}
			}
		}

		if (processImage) {
			proc.setInput(imgCopy);
			proc.process(roi);
		}
		else {
			proc.setInput(pImageInput->getImage());
		}

		key = cv::waitKey(1) & 255;

		if (key == 'q' || key == 's') {
			std::cout << "Quit\n";
			break;
		} else if (key == 'r') {
			processImage = false;
		} else if (key == 'p') {
			processImage = true;
		}
	}

	if (key != 'q') {
		std::cout << "Saving config\n";
		config.saveConfig();
	}

}

static void capture(ImageInput* pImageInput) {
	std::cout << "Capturing images into directory.\n";
	std::cout << "<Ctrl-C> to quit.\n";

	while (pImageInput->nextImage()) {
		usleep(DELAY * 1000L);
	}
}

static void writeData(ImageInput* pImageInput) {
	Config config;
	config.loadConfig();
	ImageProcessor proc(config);

	KNearestOcr ocr(config);
	if (! ocr.loadTrainingData()) {
		std::cout << "Failed to load OCR training data\n";
		return;
	}
	std::cout << "OCR training data loaded.\n";
	std::cout << "<Ctrl-C> to quit.\n";

	while (pImageInput->nextImage()) {
		proc.setInput(pImageInput->getImage());
		proc.process();

		std::string result = ocr.recognize(proc.getOutput());

		usleep(DELAY*1000L);

	}
}

void onMouseCropImage(int event, int x, int y, int f, void *param){
	switch (event) {
    case cv::EVENT_LBUTTONDOWN:
        clicked = true;
        P1.x = x;
        P1.y = y;
        P2.x = x;
        P2.y = y;
        break;
    case cv::EVENT_LBUTTONUP:
        P2.x=x;
        P2.y=y;
        clicked = false;
        break;
    case cv::EVENT_MOUSEMOVE:
        if(clicked){
        P2.x=x;
        P2.y=y;
        }
        break;
    default:
        break;
    }

    if(clicked){
        if(P1.x>P2.x){
            cropRect.x=P2.x;
            cropRect.width=P1.x-P2.x;
        }
        else{
            cropRect.x=P1.x;
            cropRect.width=P2.x-P1.x;
        }

        if(P1.y>P2.y){
            cropRect.y=P2.y;
            cropRect.height=P1.y=P2.y;
        }
        else{
            cropRect.y=P1.y;
            cropRect.height=P2.y-P1.y;
        }
    }
}

ROIBox* setROIBOX(ImageInput* pImageInput) {
	ROIBox* roi = new ROIBox();
	bool pushcrop = false;

	// Set ROI
	std::cout << ">> Select ROI box to OCR, Last ROI box will check the On/Off" << std::endl;
	while (pImageInput->nextImage()) {
		cv::Mat img, imgCopy, imgCrop;
		img = pImageInput->getImage();
		img.copyTo(imgCopy);

		cv::setMouseCallback("Select ROI", onMouseCropImage, &imgCopy);

		if (clicked) pushcrop = true;

		if (cropRect.width > 0 && clicked == false) {
			imgCrop = img(cropRect).clone();
			if (pushcrop) {
				std::cout << "push" << std::endl;
				blackBox.push_back(cropRect);
				pushcrop = false;
			}
		}
		else img.copyTo(imgCrop);

		cv::rectangle(imgCopy, P1, P2, CV_RGB(255,255,0), 2);

		cv::putText(imgCopy, "Resolution: "+std::to_string(img.cols)+"x"+std::to_string(img.rows),
				cv::Point(10,20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(60000), 1);

		cv::imshow("crop", imgCrop);
		cv::imshow("Select ROI", imgCopy);

		int key = cv::waitKey(1);
		if (key == 'q') break;
	}
	cv::destroyAllWindows();
	roi->setROIBox(blackBox);

	return roi;
}

void recordData(int cam) {
	std::cout << "Record Video" << std::endl;
	
	cv::Mat frame;
	cv::VideoCapture cap(cam);

	if(!cap.isOpened()) { std::cerr << "Camera is not opened" << std::endl; exit(1); }

	
	int fps = cap.get(cv::CAP_PROP_FPS);
	int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

	cv::VideoWriter outputVideo;
	outputVideo.open("output.avi", fourcc, fps, cv::Size(width, height), true);

	if (!outputVideo.isOpened()) { std::cerr << "Recording Initialization Error" << std::endl; exit(1); }

	while(1) {
		cap.read(frame);
		if (frame.empty()) { std::cerr << "Capture Failed" << std::endl; break;}

		imshow("Live", frame);

		outputVideo.write(frame);

		int wait = int(1.0 / fps*1000);
		if (cv::waitKey(wait) >= 0) break;
	}
}

#endif /* INCLUDE_FUNCTIONS_H_ */
