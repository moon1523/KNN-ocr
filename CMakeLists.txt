cmake_minimum_required(VERSION 2.8.9)

# Specify minimum version
set(MINIMUM_TESSERACT_VERSION 4.1.0)
set(MINIMUM_LEPTONICA_VERSION 1.74)
#set(MINIMUM_PCL_VERSION 1.11)

set(PROJECT_NAME "knnocr")
project(${PROJECT_NAME} CXX)

# Include subdirectories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Check if libraries exist
find_package(Tesseract ${MINIMUM_TESSERACT_VERSION} REQUIRED)
message("Found ${Tesseract_LIBRARIES}")
find_package(Leptonica ${MINIMUM_LEPTONICA_VERSION} REQUIRED)
message("Found ${Leptonica_LIBRARIES}")
#find_package(PCL ${MINIMUM_PCL_VERSION} REQUIRED)
#message("Found ${PCL_LIBRARIES}")
#find_package(Boost REQUIRED)
#message("found ${Boost_LIBRARIES}")
find_package(OpenCV REQUIRED)
message("found ${OpenCV_LIBRARIES}")
find_package(X11 REQUIRED)
message("found ${X11_LIBRARIES}")



# Include directories
include_directories(${Tesseract_INCLUDE_DIRS})
include_directories(${Leptonica_INCLUDE_DIRS})
#include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${X11_INCLDUE_DIRS})

FILE(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*)
FILE(GLOB HEADER_FILES ${PROJECT_SOURCE_DIR}/include/*)

# The main program
add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp ${SRC_FILES} ${HEADER_FILES})

# Link with libraries
target_link_libraries(${PROJECT_NAME} ${Leptonica_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${Tesseract_LIBRARIES})
#target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${X11_LIBRARIES})

