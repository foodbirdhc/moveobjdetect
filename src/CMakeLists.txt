CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

PROJECT(move_obj_detect)

set ( CMAKE_VERBOSE_MAKEFILE off )

set ( THIRD_PART ${CMAKE_CURRENT_SOURCE_DIR}/../third_part)

if(CMAKE_TARGET_ARCH STREQUAL x86_64)
    add_definitions(-Dx86_64)
endif()

ADD_DEFINITIONS(
-O1 -ggdb -W -Wall -std=c++11
#-fsanitize=address -fno-omit-frame-pointer -fsanitize=leak
# -Wunused-variable -Wunused-parameter -Wunused-function -Wunused
# -Wno-deprecated -Woverloaded-virtual -Wwrite-strings
# -D__WUR= -D_REENTRANT -D_FILE_OFFSET_BITS=64 -DTIXML_USE_STL
)
#-fvisibility=hidden
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -DSDK_EXPORTS -DDLC_INTERFACE_EXPORTS -DCMAKE_BUILD_TYPE=Debug")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSDK_EXPORTS -DHISI_CHIP -DDLC_INTERFACE_EXPORTS -DCMAKE_BUILD_TYPE=Debug")

if(CMAKE_TARGET_ARCH STREQUAL x86_64)
    message("use x86 ==========================================")
    set(OPENCV_VERSION "opencv_x86")
    set(OUTPUT_PATH "x86")
else()
    set(OPENCV_VERSION "opencv_aarch64")
    set(OUTPUT_PATH "aarch64")
endif()


INCLUDE_DIRECTORIES(${THIRD_PART}/include)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../include)

LINK_DIRECTORIES(${THIRD_PART}/lib/${OPENCV_VERSION}/)

message(${THIRD_PART}/lib/${OPENCV_VERSION}/)

message("1----------------------------------------------------")
FILE(GLOB src_file ${CMAKE_CURRENT_SOURCE_DIR}/*)

message(${src_file})

SET(LIB_INSTALL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../install/lib/${OUTPUT_PATH})

add_library(move_obj_detect SHARED ${src_file})

target_link_libraries(move_obj_detect opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_videoio)
target_link_libraries(move_obj_detect pthread)

install(TARGETS move_obj_detect
    LIBRARY DESTINATION ${LIB_INSTALL_PATH})


