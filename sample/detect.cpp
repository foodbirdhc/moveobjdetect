#include <iostream>
#include <unistd.h>
#include "move_obj_detect.hpp"
#include "common.h"

int main(int argc, char* argv[]) {
    std::cout << "Usage: " << argv[0] << " [video_path]" << std::endl;

    if(argc < 2) {
        std::cout << "Please input video path" << std::endl;
        return -1;
    }

    std::string video_path = argv[1];
    cv::VideoCapture cap;
    cap.open(video_path);
    if( !cap.isOpened() ) {
        std::cout << "open " << video_path << " failed." << std::endl;
        return -1;
    }

    cv::Size sWH = cv::Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
                            (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter video_writer;
    int code = video_writer.fourcc('M', 'P', '4', 'V');
    video_writer.open("./result.mp4", code, 20, sWH, true);

    MoveObjDetect move_obj_detect;

    //MRect roi_rect = {100, 100, 1200, 800};
    //move_obj_detect.SetRoi(roi_rect);

    cv::Mat frame;
    int img_id = 0;
    int save_id = 161;

    while(1) {
        cap.read(frame);
        if(frame.data == NULL) {
            break;
        }
        usleep(20*1000);
        if(img_id++ % 1 != 0) {
            continue;
        }
        //cv::cvtColor(frame, frame, cv::COLOR_RGB2GRAY);
        //cv::cvtColor(frame, frame, cv::COLOR_RGB2YUV_YV12);

        cv_image_t cv_img = {
            frame.data,
            CV_PIX_FMT_BGR888,//CV_PIX_FMT_NV21,//CV_PIX_FMT_GRAY8,
            frame.cols,
            frame.rows,
            3*frame.cols,
            {0, 0},
            NULL
        };

        std::cout << "detect run ....." << std::endl;

        std::vector<ObjInfo> result;
        std::vector<ObjInfo> drop_id;
        {
            SampleTimer track_time("Track run");
            move_obj_detect.Track(cv_img, result, drop_id);
            //move_obj_detect.Detect(cv_img, result);
        }
        cv::Mat frame_grey;
        cv::cvtColor(frame, frame_grey, cv::COLOR_RGB2GRAY);
        cv_image_t back_img;
        move_obj_detect.GetBackgroundImg(back_img);
        cv::Mat img_out = cv::Mat(back_img.height, back_img.width, CV_8UC1, back_img.data);

        for (auto res : result) {
            cv::Rect save_rect = {res.obj_rect.x + res.obj_rect.width/2 - 100, res.obj_rect.y + res.obj_rect.height/2 - 100, 200, 200};
            if(save_rect.x < 0 || save_rect.y < 0 || save_rect.x+save_rect.width > frame.cols || save_rect.y+save_rect.height > frame.rows) {
                continue;
            }
            
            cv::Mat save_img;
            cv::absdiff(img_out(save_rect), frame_grey(save_rect), save_img);
            save_img *= 2;
            save_img = 255 - save_img;

            if(save_img.data == nullptr) {
                continue;
            }
            cv::imshow("save_imgage", save_img);
            std::string image_name = std::to_string(save_id);
            image_name = "00000" + image_name;
            image_name = image_name.substr(image_name.length()-6, 6);
            image_name = "./output/" + image_name +".png";
            std::cout << "saving " << image_name << std::endl;
            if (cv::waitKey(0) == 's') { 
                cv::imwrite(image_name, save_img);
                save_id++;
            }
            cv::rectangle(frame, res.obj_rect, cv::Scalar(0, 255, 0), 1, 8, 0);
            cv::putText(frame, std::to_string(res.obj_id), cv::Point(res.obj_rect.x+res.obj_rect.width, res.obj_rect.y), cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(0, 0, 255), 1);
            cv::putText(frame, std::to_string(res.obj_rect.width), cv::Point(res.obj_rect.x, res.obj_rect.y), cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(0, 0, 255), 1);
            cv::putText(frame, std::to_string(res.obj_rect.height), cv::Point(res.obj_rect.x + res.obj_rect.width, res.obj_rect.y+res.obj_rect.height/2), cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(0, 0, 255), 1);
        }

        for (auto dro : drop_id) {
            LOG_INFO("drop id : " << dro.obj_id);
        }

        video_writer.write(frame);
    #ifdef x86_64
        cv::imshow("window", frame);
        cv::waitKey(100);
    #endif
    }

    video_writer.release();

    return 0;
}
