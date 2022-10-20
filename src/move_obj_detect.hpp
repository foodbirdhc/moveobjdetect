#ifndef __MOVE_OBJ_DETECT_HPP__
#define __MOVE_OBJ_DETECT_HPP__

#include <iostream>
#include <vector>
#include <map>
#include "opencv2/opencv.hpp"

#define LOG_INFO(log)   std::cout << "[INFO] " << log << std::endl

#define LOG_ERROR(log)   std::cerr << "[ERROR] " << log << std::endl

typedef struct Timer {
    long int tv_sec; //!< second
    long int tv_usec; //!< microsecond
} Timer;

typedef struct MRect {
    int x;
    int y;
    int width;
    int height;
} MRect;

typedef enum DetectMode {
    MOVE_MODE,
    STATIC_MODE,
    INVALID_MODE
} DetectMode;

typedef struct StaticObj {
    int obj_id;
    cv::Rect obj_rect;
    cv::Point center_p;
    int offset_thr;
    Timer first_time;
    unsigned long static_thr; // s
} StaticObj;

/// cv pixel format definition
typedef enum cv_pixel_format_e {
        CV_PIX_FMT_GRAY8, ///< Y    1       8bpp ( 单通道8bit灰度像素 )
        CV_PIX_FMT_YUV420P, ///< YUV  4:2:0   12bpp ( 3通道, 一个亮度通道,
        ///< 另两个为U分量和V分量通道, 所有通道都是连续的 )
        CV_PIX_FMT_NV12, ///< YUV  4:2:0   12bpp ( 2通道, 一个通道是连续的亮度通道,
        ///< 另一通道为UV分量交错 )
        CV_PIX_FMT_NV21, ///< YUV  4:2:0   12bpp ( 2通道, 一个通道是连续的亮度通道,
        ///< 另一通道为VU分量交错 )
        CV_PIX_FMT_BGRA8888, ///< BGRA 8:8:8:8 32bpp ( 4通道32bit BGRA 像素 )
        CV_PIX_FMT_BGR888, ///< BGR  8:8:8   24bpp ( 3通道24bit BGR 像素 )
        CV_PIX_FMT_GRAY16,
        CV_PIX_FMT_NONE
} cv_pixel_format_e;

/// 图像格式定义
typedef struct cv_image_t {
        unsigned char *data; ///< 图像数据指针
        cv_pixel_format_e pixel_format; ///< 像素格式
        int width; ///< 宽度(以像素为单位)
        int height; ///< 高度(以像素为单位)
        int stride; ///< 跨度, 即每行所占的字节数
        Timer time_stamp; ///< 时间戳
        void *reserved; ///< 内部保留, 请勿使用
} cv_image_t;

typedef struct ObjInfo {
    int obj_id;
    cv::Rect obj_rect;
} ObjInfo;

typedef struct DroObjInfo {
    ObjInfo obj_info;
    int drop_count;
    int invalid_count;
} DroObjInfo;

constexpr int OBJ_OFFSET_THR = 900; // 30^2
constexpr int CHANGE_BACKGROUND_THR = 200;
constexpr int DROP_LENGTH = 4;
constexpr int UPDATE_BACKGROUND_THR = 48;
constexpr int OBJ_NOMOVE_OFFSET_THR = 2; // 2^2
constexpr double MAX_DIFF_THR = 20.;

class MoveObjDetect {
 public:
    MoveObjDetect();

    ~MoveObjDetect();

    void Init();

    void Detect(const cv_image_t& img_in, std::vector<ObjInfo>& result);

    void Track(const cv_image_t& img_in, std::vector<ObjInfo>& result, std::vector<ObjInfo>& drop_id);

    void GetBackgroundImg(cv_image_t& img_out);

    void SetDetectMode(const DetectMode mode);

    const DetectMode GetDetectMode();

    void SetRoi(const MRect& roi);

    const MRect GetRoi();

    int SetBackGroundImg(const cv_image_t& img_in);

    void SetObjOffsetThr(const int thr);

    const int GetObjOffsetThr() const;

    void SetChangeBackgroundThr(const int thr);

    const int GetChangeBackgroundThr() const;

    void SetSmoothImgThr(const int thr);

    const int GetSmoothImgThr() const;

    void SetObjMinSize(const int size);

    const int GetObjMinSize() const;

    void SetObjMaxSize(const int size);

    const int GetObjMaxSize() const;

 private:
    int BaseDetect(const cv_image_t& img_in, std::vector<ObjInfo>& result);

    void allImageToGreyImg(const cv_image_t& img_in, cv::Mat & img_out);

    bool CheckIfOverOffset(const cv::Point& p1, const cv::Point& p2, const int offset_thr);

    bool CheckIfOverOffset(const cv::Point& p1, const std::vector<StaticObj>& obj_static, const int offset_thr);

    void AdujstRoi(const cv::Mat& img_in);

    void UpdateBackGround(const cv::Mat &cur_img, const cv::Rect& roi_rect);

    void MoveTargetMode(const cv::Mat& run_img, const cv::Rect& cur_rect);

 private:
    cv::Rect roi_;
    cv::Mat background_img_;
    //cv::Mat element_;
    cv::Mat diff_img;
    DetectMode detect_mode_;
    double thr_;
    int obj_id_;
    std::vector<StaticObj> obj_static_;
    std::vector<DroObjInfo> last_obj_info_;
    std::vector<DroObjInfo> static_obj_info_;
    int obj_size_min_size_;
    int obj_size_max_size_;
    int smooth_img_thr_;
    int smooth_img_count_;
    int obj_offset_thr_;
    int change_background_thr_;
    int noobj_count_;
    int over_max_diff_count_;
};

#endif


