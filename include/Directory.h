#ifndef INCLUDE_DIRECTORY_H_
#define INCLUDE_DIRECTORY_H_

#include <string>
#include <list>

class Directory {
public:
	Directory(const char* path, const char* extension);

	std::list<std::string> list();
	std::string fullpath(const std::string filename);

private:
	bool hasExtension(const char* name, const char* ext);

	std::string _path;
	std::string _extension;
};

#endif /* INCLUDE_DIRECTORY_H_ */
