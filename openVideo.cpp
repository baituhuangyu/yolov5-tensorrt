//
// Created by 羽黄 on 2020/7/8.
//

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace cv;
using namespace std;


int main()
{
    cout << "Built with OpenCV " << CV_VERSION << endl;

    VideoCapture video;
    video.open("video.avi");// 打开视频文件
    cout << "frame count: " << video.get(cv::CAP_PROP_FRAME_COUNT);

    if(!video.isOpened())// 判断是否打开成功
    {
        cout << "open video file failed. " << endl;
        return -1;
    }

    while(true)
    {
        Mat frame;
        video >> frame;// 读取图像帧至frame
        if(!frame.empty())    // frame是否为空
        {
            imshow("video", frame);// 显示图像

//            cout << frame;
        }

        if(waitKey(30) > 0)        // delay 30 ms 等待是否按键
        {
            break;
        }
        if (!frame.isContinuous()) {
            video.release();
            break;
        }
    }

    return 0;
}

