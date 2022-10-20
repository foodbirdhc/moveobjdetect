CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

PROJECT(test_sample)

set ( CMAKE_VERBOSE_MAKEFILE off )

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

set ( THIRD_PART ${CMAKE_CURRENT_SOURCE_DIR}/third_part)

INCLUDE_DIRECTORIES(${THIRD_PART}/include)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src/)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include/)

LINK_DIRECTORIES(${THIRD_PART}/lib/${OPENCV_VERSION})
LINK_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/install/lib/${OUTPUT_PATH})

message(${THIRD_PART}/lib/${OPENCV_VERSION}/)

message("1----------------------------------------------------")
#FILE(GLOB src_file ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

SET(src_file
    ${CMAKE_CURRENT_SOURCE_DIR}/sample/detect.cpp
)
message("2----------------------------------------------------")

message(${src_file})

SET(SAMPLE_INSTALL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/install/bin)

foreach(src ${src_file})
    message("building " ${src} " -----------------------------------------")
    string(REGEX MATCH "[^/]+$" temp_src_file ${src})
    string(REPLACE ".cpp" "" exe_file_name ${temp_src_file})

    message(${exe_file_name} ${src})

    add_executable(${exe_file_name} ${src})

    target_link_libraries(${exe_file_name} move_obj_detect)

    #target_link_libraries(${exe_file_name} opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_videoio avcodec avformat avformat swscale)
    target_link_libraries(${exe_file_name} opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_videoio)
    target_link_libraries(${exe_file_name} pthread)

    install(TARGETS ${exe_file_name}
        RUNTIME DESTINATION ${SAMPLE_INSTALL_PATH})
endforeach()

install(DIRECTORY ${THIRD_PART}/lib/${OPENCV_VERSION}
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/third_part)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/data)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample/env.sh
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/bin)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/sample/
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/sample)


