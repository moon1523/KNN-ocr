#include <string>
#include <list>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "ImageInput.h"
#include "ImageProcessor.h"
#include "KNearestOcr.h"
#include "functions.h"

static void usage(const char* progname) {
    std::cout << "Program to read and recognize the C-arm screen data with OpenCV.\n";
    std::cout << "Usage: " << progname << " [-i <dir>|-c <cam>] [-l|-t|-a|-w|-o <dir>] [-s <delay>] [-v <level>\n";
    std::cout << "\nImage input:\n";
    std::cout << "  -i <image directory> : read image files (png) from directory.\n";
    std::cout << "  -c <camera number> : read images from camera.\n";
    std::cout << "\nOperation:\n";
    std::cout << "  -a : adjust camera.\n";
    std::cout << "  -o <directory> : capture images into directory.\n";
    std::cout << "  -l : learn OCR.\n";
    std::cout << "  -t : test OCR.\n";
    std::cout << "  -w : write OCR data to RR database. This is the normal working mode.\n";
    std::cout << "\nOptions:\n";
    std::cout << "  -s <n> : Sleep n milliseconds after processing of each image (default=1000).\n";
    std::cout << "  -v <l> : Log level. One of DEBUG, INFO, ERROR (default).\n";
}

int main(int argc, char** argv) {

	int opt;
	ImageInput* pImageInput = 0;
	int inputCount = 0;
	std::string outputDir;
	std::string logLevel = "ERROR";
	char cmd = 0;
	int  cmdCount = 0;
	int  cam = 0;

	while ((opt = getopt(argc, argv, "i:c:ltaws:o:v:h:r")) != -1) {
		switch (opt) {
			case 'i':
				pImageInput = new DirectoryInput(Directory(optarg, ".png"));
				inputCount++;
				break;
			case 'c':
				pImageInput = new CameraInput(atoi(optarg));
				inputCount++;
				cam = atoi(optarg);
				break;
			case 'l':
			case 't':
			case 'a':
			case 'w':
				cmd = opt;
				cmdCount++;
				break;
			case 'o':
				cmd = opt;
				cmdCount++;
				outputDir = optarg;
				break;
			case 's':
				DELAY = atoi(optarg);
				break;
			case 'v':
				logLevel = optarg;
				break;
			case 'h':
			default:
				usage(argv[0]);
				exit(EXIT_FAILURE);
				break;
		}
	}
	if (inputCount != 1) {
		std::cerr << "*** You should specify exactly one camera or input directory!\n\n";
		usage(argv[0]);
		exit(EXIT_FAILURE);
	}
	if (cmdCount != 1) {
		std::cerr << "*** You should specify exactly one operation!\n\n";
		usage(argv[0]);
		exit(EXIT_FAILURE);
	}

	switch (cmd) {
		case 'o':
			pImageInput->setOutputDir(outputDir);
			capture(pImageInput);
			break;
		case 'l':
			learnOcr(pImageInput);
			break;
		case 't':
			testOcr(pImageInput, cam);
			break;
		case 'a':
			adjustCamera(pImageInput);
			break;
		case 'w':
			writeData(pImageInput);
			break;
	}

	delete pImageInput;
	exit(EXIT_SUCCESS);
}