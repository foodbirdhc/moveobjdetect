#include "move_obj_detect.hpp"
#include "common.h"
//#include "opencv2/features2d.hpp"

MoveObjDetect::MoveObjDetect() : detect_mode_(MOVE_MODE),
                                 thr_(0.),
                                 obj_id_(0),
                                 obj_size_min_size_(64),
                                 obj_size_max_size_(512),
                                 smooth_img_thr_(1),
                                 smooth_img_count_(0),
                                 obj_offset_thr_(OBJ_OFFSET_THR),
                                 change_background_thr_(CHANGE_BACKGROUND_THR), 
                                 noobj_count_(0),
                                 over_max_diff_count_(0) {
    Init();

}

MoveObjDetect::~MoveObjDetect() {

}

void MoveObjDetect::Init() {
    roi_.x = 0;
    roi_.y = 0;
    roi_.width = 1920;
    roi_.height = 1080;
}

void MoveObjDetect::SetDetectMode(const DetectMode mode) {
    detect_mode_ = mode;
}

const DetectMode MoveObjDetect::GetDetectMode() {
    return detect_mode_;
}

void MoveObjDetect::SetRoi(const MRect& roi) {
    roi_.x = roi.x;
    roi_.y = roi.y;
    roi_.width = roi.width;
    roi_.height = roi.height;
}

const MRect MoveObjDetect::GetRoi() {
    MRect roi_rect = {roi_.x, roi_.y, roi_.width, roi_.height};
    return roi_rect;
}

void MoveObjDetect::SetObjOffsetThr(const int thr) {
    obj_offset_thr_ = thr;
}

const int MoveObjDetect::GetObjOffsetThr() const {
    return obj_offset_thr_;
}

void MoveObjDetect::SetChangeBackgroundThr(const int thr) {
    change_background_thr_ = thr;
}

const int MoveObjDetect::GetChangeBackgroundThr() const {
    return change_background_thr_;
}

void MoveObjDetect::SetSmoothImgThr(const int thr) {
    smooth_img_thr_ = thr;
}

const int MoveObjDetect::GetSmoothImgThr() const {
    return smooth_img_thr_;
}

void MoveObjDetect::SetObjMinSize(const int size) {
    obj_size_min_size_ = size;
}

const int MoveObjDetect::GetObjMinSize() const {
    return obj_size_min_size_;
}

void MoveObjDetect::SetObjMaxSize(const int size) {
    obj_size_max_size_ = size;
}

const int MoveObjDetect::GetObjMaxSize() const {
    return obj_size_max_size_;
}

void MoveObjDetect::AdujstRoi(const cv::Mat& img_in) {
    if (roi_.x < 0) {
        roi_.x = 0;
    }

    if (roi_.y < 0) {
        roi_.y = 0;
    }

    if ((roi_.x + roi_.width) > img_in.cols || roi_.width == 0) {
        roi_.width = img_in.cols - roi_.x;
    }

    if ((roi_.y + roi_.height) > img_in.rows || roi_.height == 0) {
        roi_.height = img_in.rows - roi_.y;
    }
}

int MoveObjDetect::SetBackGroundImg(const cv_image_t& img_in) {
    if(img_in.data == NULL) {
        LOG_ERROR("image is null");
        return -1;
    }

    {
        SampleTimer track_time("allImageToGreyImg run");
        allImageToGreyImg(img_in, background_img_);
    }

    obj_size_min_size_ = background_img_.cols / obj_size_min_size_;

    AdujstRoi(background_img_);

    background_img_ = background_img_(roi_).clone();

    cv::Mat temp_img;
    thr_ = cv::threshold(background_img_, temp_img, 0, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);
    LOG_INFO("thr set : " << thr_);

    //element_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(obj_size_min_size_, obj_size_min_size_));

    LOG_INFO("<=== Set background image finished ===>");
    return 1;
}

void MoveObjDetect::GetBackgroundImg(cv_image_t& img_out) {

    img_out.data = background_img_.data;
    img_out.pixel_format = CV_PIX_FMT_GRAY8;
    img_out.width = background_img_.cols;
    img_out.height = background_img_.rows;
    img_out.stride = background_img_.cols;
    img_out.reserved = NULL;

    return;
}

void MoveObjDetect::UpdateBackGround(const cv::Mat &cur_img, const cv::Rect& roi_rect) {
    cv::Mat target_roi = background_img_(roi_rect);
    cur_img(roi_rect).copyTo(target_roi);
}

void MoveObjDetect::Detect(const cv_image_t& img_in, std::vector<ObjInfo>& result) {
    BaseDetect(img_in, result);
}

void MoveObjDetect::Track(const cv_image_t& img_in, std::vector<ObjInfo>& result, std::vector<ObjInfo>& drop_id) {
    BaseDetect(img_in, result);

    for(auto & last_obj : last_obj_info_) {
        last_obj.drop_count++;
    }

    for(auto & new_obj : result) {
        int valid_id = obj_id_;
        int min_offset = obj_offset_thr_;
        for(auto & last_obj : last_obj_info_) {
            cv::Point new_p = {new_obj.obj_rect.x + new_obj.obj_rect.width/2, new_obj.obj_rect.y + new_obj.obj_rect.height/2};
            cv::Point last_p = {last_obj.obj_info.obj_rect.x + last_obj.obj_info.obj_rect.width/2, last_obj.obj_info.obj_rect.y + last_obj.obj_info.obj_rect.height/2};
            int cur_offset = (new_p.x - last_p.x)*(new_p.x - last_p.x) +\
                             (new_p.y - last_p.y)*(new_p.y - last_p.y);
            //LOG_INFO("------- cur_offset : " << cur_offset);
            if(cur_offset < min_offset) {
                min_offset = cur_offset;
                valid_id = last_obj.obj_info.obj_id;
            }
        }
        if(min_offset < obj_offset_thr_) {
            new_obj.obj_id = valid_id;
            for(auto & last_obj : last_obj_info_) {
                if(last_obj.obj_info.obj_id == valid_id) {
                    last_obj.drop_count = 0;
                }
            }
        } else {
            new_obj.obj_id = obj_id_++;
        }
    }

    for(auto iter = last_obj_info_.begin(); iter != last_obj_info_.end(); ) {
        if(iter->drop_count >= DROP_LENGTH) {
            drop_id.push_back(iter->obj_info);
            iter = last_obj_info_.erase(iter);
            continue;
        }
        ++iter;
    }

    for(auto new_obj : result) {
        int index = 0;
        for( ; index < last_obj_info_.size(); ++index) {
            if(new_obj.obj_id == last_obj_info_[index].obj_info.obj_id) {
                last_obj_info_[index].obj_info = new_obj;
                break;
            }
        }
        if(index >= last_obj_info_.size()) {
            DroObjInfo drop_info;
            drop_info.obj_info = new_obj;
            drop_info.drop_count = 0;
            last_obj_info_.push_back(drop_info);
        }
    }

    //LOG_INFO("last_obj_info_ size : " << last_obj_info_.size());
    // obj id track ....
}

int MoveObjDetect::BaseDetect(const cv_image_t& img_in, std::vector<ObjInfo>& result) {
    if(img_in.data == NULL) {
        LOG_ERROR("image is null");
        return -1;
    }

    if(background_img_.data == NULL) {
        SetBackGroundImg(img_in);
    }

    cv::Mat run_img;
    {
        SampleTimer track_time("allImageToGreyImg run");
        allImageToGreyImg(img_in, run_img);
    }
    ++smooth_img_count_;

    run_img = run_img(roi_); // 裁剪出roi区域做算法
    
    do {
        { // 最近图像和背景做差并进行拉伸
            SampleTimer track_time("cv::absdiff run");
            cv::absdiff(run_img, background_img_, diff_img); // 计算差异图
            cv::Scalar full_frame_diff_mean_val = cv::mean(diff_img);
            auto valid_full_diff_mean = full_frame_diff_mean_val.val[0];
            if(valid_full_diff_mean >= MAX_DIFF_THR) {
                over_max_diff_count_++;
                if(over_max_diff_count_ >= 20) {
                    LOG_INFO("change background image because over max diff .... ");
                    SetBackGroundImg(img_in);
                    smooth_img_count_ = 0;
                    over_max_diff_count_ = 0;
                }
                continue;
            } else if(over_max_diff_count_ > 0) {
                --over_max_diff_count_;
            }

            // 本来这里可以插入一些滤波模块，但是尝试了多种滤波，效果反而变差了，而且更耗时
            diff_img *= 2; // 拉伸图片，突出差异部分
            cv::Mat temp_img = (255 - diff_img);
            cv::imshow("big_map", temp_img);
            if(cv::waitKey(0) == 'y') {
                
                cv::imwrite("test_image.jpg", temp_img);
            }
        }

        { // 对拉伸的差值图进行二值化计算，但阈值采用背景的自适应二值化阈值
            SampleTimer track_time("cv::threshold run");
            cv::threshold(diff_img, diff_img, thr_, 255, cv::THRESH_BINARY); // 对拉伸后的差异图进行二值计算
        }

        std::vector<std::vector<cv::Point> > contours; // Vector for storing contours
        { // 对目标进行膨胀计算，但opencv自带膨胀接口太耗时，所以采用新的实现方法
            //因为该模块非常耗时，所以用其他方式重写
            SampleTimer track_time("dilate run");
            //element_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(obj_size_min_size_, obj_size_min_size_));
            //cv::dilate(diff_img, diff_img, element_, cv::Point(-1, -1), 2, cv::BORDER_CONSTANT);
            //cv::erode(diff_img, diff_img, element, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT);

            cv::findContours( diff_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE ); // Find the contours in the image
            for(auto contour : contours) {
                auto cur_rect = cv::boundingRect(contour);
                cv::Point c_p = {cur_rect.x + cur_rect.width/2, cur_rect.y + cur_rect.height/2}; // find center point

                if(cur_rect.width > obj_size_max_size_ || cur_rect.height > obj_size_max_size_) {
                    LOG_INFO("obj is too max");
                    continue;
                }

                // 过滤选取背景时有些静物微小变化的影响
                bool if_over_offset = CheckIfOverOffset(c_p, obj_static_, obj_offset_thr_);
                if( !if_over_offset ) {
                    continue;
                }

                // 进行目标膨胀处理，可以合并目标周围噪点，关键处理
                LOG_INFO("obj_size_min_size_ : " << obj_size_min_size_);
                //obj_size_min_size_ = 30;
                int i = (cur_rect.x - obj_size_min_size_) >= 0? (cur_rect.x - obj_size_min_size_) : 0;
                for(; i < (cur_rect.x + cur_rect.width + 2*obj_size_min_size_) && i < diff_img.cols; ++i) {
                    int j = (cur_rect.y - obj_size_min_size_) >= 0? (cur_rect.y - obj_size_min_size_): 0;
                    for(; j < (cur_rect.y + cur_rect.height + obj_size_min_size_) && j < diff_img.rows; ++j) {
                        diff_img.data[j*diff_img.cols + i] = 255;
                    }
                }
            }
        }

        {
            SampleTimer track_time("cv::findContours run");
            contours.clear();
            cv::findContours( diff_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE ); // Find the contours in the image
        }

        int cur_frame_obj_id = 0;
        for(auto contour : contours) {
            //double area = cv::contourArea(contour);  //  Find the area of contour
            auto cur_rect = cv::boundingRect(contour);
            cv::Point c_p = {cur_rect.x + cur_rect.width/2, cur_rect.y + cur_rect.height/2}; // find center point

            int index = 0;
            if(smooth_img_count_ < smooth_img_thr_) {
                bool if_over_offset = CheckIfOverOffset(c_p, obj_static_, obj_offset_thr_);
                if(if_over_offset) {
                    LOG_INFO(">>>>>>> push a static obj");
                    StaticObj new_obj = {cur_frame_obj_id++, cur_rect, c_p, obj_offset_thr_, {0, 0}, 3};
                    obj_static_.push_back(new_obj);
                }

                noobj_count_ = 0;
                continue;
            }
            
            // 过滤选取背景时有些静物微小变化的影响
            bool if_over_offset = CheckIfOverOffset(c_p, obj_static_, obj_offset_thr_);
            if( !if_over_offset ) {
                continue;
            }

            {
                SampleTimer track_time("filter run");
                //cv::Rect small_img_rect = {cur_rect.x + cur_rect.width/4, cur_rect.y + cur_rect.height/4, cur_rect.width/2, cur_rect.height/2};
                cv::Mat back_img_roi = background_img_(cur_rect);
                cv::Mat cur_img_roi = run_img(cur_rect);

                cv::Scalar mean_val = 0;
                mean_val = cv::mean(back_img_roi);
                auto back_mean = mean_val.val[0];
                mean_val = cv::mean(cur_img_roi);
                auto cur_mean = mean_val.val[0];
                cv::Mat small_diff_img;
                if(back_mean > cur_mean) {
                    small_diff_img = back_img_roi - cur_img_roi;
                } else {
                    small_diff_img = cur_img_roi - back_img_roi;
                }

                cv::Mat abs_diff_img;
                cv::absdiff(cur_img_roi, back_img_roi, abs_diff_img); // 计算差异图;
                abs_diff_img += small_diff_img;
                //std::cout << "----------------------------------------------------------------------" << std::endl;
                //for(int h = 0; h < small_diff_img.rows; ++h) {
                //    for(int w = 0; w < small_diff_img.cols; ++w) {
                //        std::cout << std::setw(2) << (int)small_diff_img.data[h*small_diff_img.cols+w] << " ";
                //    }
                //    std::cout << std::endl;
                //}
                //std::cout << "----------------------------------------------------------------------" << std::endl;

                //cv::imshow("abs_diff_img", abs_diff_img);
                //cv::waitKey(0);

                //adaptiveThreshold(~abs_diff_img, abs_diff_img, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 31, 10);
                cv::threshold(abs_diff_img, abs_diff_img, 0, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);

                std::vector<std::vector<cv::Point> > small_contours; // Vector for storing contours
                cv::findContours( abs_diff_img, small_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE ); // Find the contours in the image
                int total_area = 0;
                for(auto small_con : small_contours) {
                    total_area += cv::contourArea(small_con);
                }

                double area_rate = (double)total_area/(double)(cv::contourArea(contour));
                //double area_rate = (double)total_area/(double)(small_img_rect.width*small_img_rect.height);
                LOG_INFO("area_rate : " << area_rate);
                if(area_rate < 0.4/(double)obj_size_min_size_) {
                    continue;
                }

                //mean_val = cv::mean(abs_diff_img);
                //float valid_mean = mean_val.val[0];
                //mean_val = cv::mean(back_img_roi);
                //float mean_thr = (float)back_mean; //动态阈值生成，大致原理是，背景越黑，和老鼠颜色越相近，差异会越小
                //LOG_INFO("[TEST] Mean thr " << mean_thr << "  valid_mean = " << valid_mean);
                //if(std::abs(valid_mean) < 12) { // 设置局部差异过滤，可以去除原有物体位置小移动，或者小物体出现的影响
                //    continue;
                //}
            }

            if(detect_mode_ == MOVE_MODE) {
                MoveTargetMode(run_img, cur_rect);
            }

            // find valid obj
            noobj_count_ = 0;
            cur_rect.x += roi_.x;
            cur_rect.y += roi_.y;
            ObjInfo obj_info = {cur_frame_obj_id++, cur_rect};
            result.push_back(obj_info);

        }

        if(result.empty()) {
            noobj_count_++;
            if(noobj_count_ >= change_background_thr_) {
                LOG_INFO("change background image .... ");
                SetBackGroundImg(img_in);
                smooth_img_count_ = 0;
            }
        }
    } while(false);

    if(detect_mode_ == MOVE_MODE) {
        auto iter = static_obj_info_.begin();
        for(; iter != static_obj_info_.end();) {
            iter->invalid_count++;
            if(iter->invalid_count >= 2) {
                iter = static_obj_info_.erase(iter);
                continue;
            }
            ++iter;
        }
    }

    //cv::imshow("back", background_img_);
    //cv::waitKey(10);
 
    return 0;
}

void MoveObjDetect::MoveTargetMode(const cv::Mat& run_img, const cv::Rect& cur_rect) {
    auto iter = static_obj_info_.begin();
    bool need_push = true;
    cv::Point p2 = {cur_rect.x + cur_rect.width / 2,
                    cur_rect.y + cur_rect.height / 2};
    for(; iter != static_obj_info_.end(); ++iter) {
        cv::Point p1 = {iter->obj_info.obj_rect.x + iter->obj_info.obj_rect.width / 2,
                        iter->obj_info.obj_rect.y + iter->obj_info.obj_rect.height / 2};
        auto is_over = CheckIfOverOffset(p1, p2, OBJ_NOMOVE_OFFSET_THR);
        if(!is_over) {
            iter->drop_count++;
            if(iter->drop_count >= UPDATE_BACKGROUND_THR) {
                //std::cout << "=============== UpdateBackGround =========================" << std::endl;
                UpdateBackGround(run_img, cur_rect);
                static_obj_info_.erase(iter);
                need_push = false;
                break;
            }
            iter->obj_info.obj_rect = cur_rect;
            iter->invalid_count = 0;
            break;
        }
    }
    if(need_push) {
        DroObjInfo static_obj;
        static_obj.invalid_count = 0;
        static_obj.drop_count = 1;
        static_obj.obj_info.obj_rect = cur_rect;
        static_obj_info_.emplace_back(static_obj);
    }

    //std::cout << "static_obj_info_ size : " << static_obj_info_.size() << std::endl;
}

bool MoveObjDetect::CheckIfOverOffset(const cv::Point& p1, const cv::Point& p2, const int offset_thr) {
    // 过滤选取背景时有些静物微小变化的影响
    auto offset_val = (p1.x - p2.x)*(p1.x - p2.x) + \
                        (p1.y - p2.y)*(p1.y - p2.y);
    //std::cout << "offset_val ==== " << offset_val << std::endl;
    if(offset_val < offset_thr) {
        return false;
    }
    return true;
}

bool MoveObjDetect::CheckIfOverOffset(const cv::Point& p1, const std::vector<StaticObj>& obj_static, const int offset_thr) {
    // 过滤选取背景时有些静物微小变化的影响
    int index = 0;
    for(index = 0; index < obj_static_.size(); ++index) {
        auto offset_val = (p1.x - obj_static[index].center_p.x)*(p1.x - obj_static[index].center_p.x) + \
                            (p1.y - obj_static[index].center_p.y)*(p1.y - obj_static[index].center_p.y);
        if(offset_val < offset_thr) {
            break;
        }
    }
    if(index < obj_static_.size()) {
        return false;
    }

    return true;
}

void MoveObjDetect::allImageToGreyImg(const cv_image_t& img_in, cv::Mat & img_out) {
    switch (img_in.pixel_format) {
        case CV_PIX_FMT_GRAY8: {
            //LOG_INFO("image format is CV_PIX_FMT_GRAY8");
            img_out = cv::Mat(img_in.height, img_in.width, CV_8UC1, img_in.data);
            break;
        }

        case CV_PIX_FMT_YUV420P:
        case CV_PIX_FMT_NV12:
        case CV_PIX_FMT_NV21: {
            //LOG_INFO("image format is CV_PIX_FMT_YUV");
            img_out = cv::Mat(2*img_in.height/3, img_in.width, CV_8UC1, img_in.data);
            //cv::cvtColor(img_out, img_out, cv::COLOR_YUV2GRAY_420);
            break;
        }

        case CV_PIX_FMT_BGR888: {
            //LOG_INFO("image format is CV_PIX_FMT_BGR888");
            img_out = cv::Mat(img_in.height, img_in.width, CV_8UC3, img_in.data);
            cv::cvtColor(img_out, img_out, cv::COLOR_BGR2GRAY);
            break;
        }

        default:
            break;
    }
}
