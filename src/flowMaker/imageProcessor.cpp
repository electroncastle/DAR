#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

// Without the std namespace here the opencv gpu module will complain about
// non existing vector<>
using namespace std;
//#include "opencv2/gpu/gpu.hpp"

#include <stdio.h>
#include <iostream>
using namespace cv;
//using namespace cv::gpu;

// Opoencv 3
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"


#include <QCoreApplication>
#include <QtAlgorithms>
#include <QDirIterator>
#include <QDebug>
#include <QDir>
#include <extractFlow.h>

#define USE_MPI 1

#ifdef USE_MPI
    #include <mpi/mpi.h>
#endif

#include "imageProcessor.h"
using namespace cv::cuda;

//int myid, numprocs;

inline bool isFlowCorrect(Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

static Vec3b computeColor(float fx, float fy)
{
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static Vec3i colorWheel[NCOLS];

    if (first)
    {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float) CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    Vec3b pix;

    for (int b = 0; b < 3; b++)
    {
        const float col0 = colorWheel[k0][b] / 255.0f;
        const float col1 = colorWheel[k1][b] / 255.0f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(255.0 * col);
    }

    return pix;
}

void drawOpticalFlow(const Mat_<float>& flowx, const Mat_<float>& flowy, Mat& dst, float maxmotion)
{
    dst.create(flowx.size(), CV_8UC3);
    dst.setTo(Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flowx.rows; ++y)
        {
            for (int x = 0; x < flowx.cols; ++x)
            {
                Point2f u(flowx(y, x), flowy(y, x));

                if (!isFlowCorrect(u))
                    continue;

                maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }

    for (int y = 0; y < flowx.rows; ++y)
    {
        for (int x = 0; x < flowx.cols; ++x)
        {
            Point2f u(flowx(y, x), flowy(y, x));

            if (isFlowCorrect(u))
                dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}

void showFlow(const char* name, const cv::cuda::GpuMat& d_flow)
{
    GpuMat planes[2];
    cuda::split(d_flow, planes);

    Mat flowx(planes[0]);
    Mat flowy(planes[1]);

    Mat out;
    drawOpticalFlow(flowx, flowy, out, 10);

    imshow(name, out);
}

static void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y,
       double lowerBound, double higherBound)
{
    #define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : \
        cvRound(255*((v) - (L))/((H)-(L))))
	for (int i = 0; i < flow_x.rows; ++i) {
		for (int j = 0; j < flow_y.cols; ++j) {
			float x = flow_x.at<float>(i,j);
			float y = flow_y.at<float>(i,j);
			img_x.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
			img_y.at<uchar>(i,j) = CAST(y, lowerBound, higherBound);
		}
	}
	#undef CAST
}

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,double, const Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}



static bool jobComp(JobItem* left, JobItem *right)
{
  return left->filename.compare(right->filename) < 0;
}


cv::Mat thresholdAbout(cv::Mat m, int val)
{
    double minval, maxval;

    cv::Mat o;
    m.convertTo(o, CV_16SC1, 1, -val);
    cv::minMaxLoc(o, &minval, &maxval);
    float v1 = -127.0/minval;
    float v2 = 127.0/maxval;
    if (v2<v1) v1=v2;

    m = o*v1+val;

    cv::minMaxLoc(m, &minval, &maxval);
    m.convertTo(o, CV_8UC1, 1, 0);
    return o;
}

int makeRFBFlowImage(QString outPath, QString videoFilename)
{
    QStringList toks = videoFilename.split("/");
    QString videoName = toks[toks.size()-1];

    QFileInfo fi(videoName);
    QStringList pathToks = fi.baseName().split("_");

//    if (fi.baseName() != "v_YoYo_g25_c03"){
//        return 0;
//    }

    QString imgPath = outPath+"/"+pathToks[1]+"/"+fi.baseName();

    QDirIterator it(imgPath.toStdString().c_str(), QDirIterator::Subdirectories);
    while (it.hasNext()) {
        QString filename =  it.next();
        if (!it.fileInfo().isFile()){
            continue;
        }

        QStringList toks = it.fileInfo().baseName().split("_");
        if (toks.size()<2)
            continue;

        if (toks[0] != "image")
            continue;

        //if (toks[1] != "0030")  continue;

        QString xflow = it.fileInfo().path()+"/flow_x_"+toks[1]+".jpg";
        QString yflow = it.fileInfo().path()+"/flow_y_"+toks[1]+".jpg";
        QString outFile = it.fileInfo().path()+"/flow_"+toks[1]+".jpg";

        // Make the RGB image
        cv::Mat xof = cv::imread(xflow.toStdString(), IMREAD_GRAYSCALE);
        cv::Mat yof = cv::imread(yflow.toStdString(), IMREAD_GRAYSCALE);

//        cv::normalize(xof,xof,0,255,cv::NORM_MINMAX);
//        cv::normalize(yof,yof,0,255,cv::NORM_MINMAX);

//#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : \
//    cvRound(255*((v) - (L))/((H)-(L))))

        xof = thresholdAbout(xof, 128);
        yof = thresholdAbout(yof, 128);

        cv::Mat xoff, yoff;
        double minval, maxval;
        cv::minMaxLoc(xof, &minval, &maxval);

        xof.convertTo(xoff, CV_32F, 2/255.0, -1);

        cv::minMaxLoc(yof, &minval, &maxval);
        yof.convertTo(yoff, CV_32F, 2/255.0, -1);

        cv::Mat magf;
        cv::magnitude(xoff, yoff, magf);

        cv::Mat mag = cv::Mat(xoff.rows, xoff.cols, CV_8UC1, cv::Scalar(128));
        magf.convertTo(mag, CV_8UC1, 127.0, 128);
        cv::minMaxLoc(mag, &minval, &maxval);

        //cv::normalize(mag,mag,0,255,cv::NORM_MINMAX);
        //cv::convertScaleAbs(mag,mag);

        std::vector<cv::Mat> channels;
        channels.push_back(mag);
        channels.push_back(yof);
        channels.push_back(xof);

        cv::Mat out;
        cv::merge(channels, out);
        cv::imwrite(outFile.toStdString(), out);
        std::cout << outFile.toStdString().c_str() << std::endl << std::flush;

//        cv::imshow("Main", xof); cv::waitKey(0);
//        cv::imshow("Main", out); cv::waitKey(0);

    }

    return 1;
}



void ImageProcessor::finish()
{
    isRunning = 0;
}


int ImageProcessor::process(QString videoClass, QString videoFileName)
{
    bool overwrite = false;
    int step = 1;
    int type = 1;
    int bound = 20;

    QFileInfo fileInfo(videoFileName);
    QString classPath = path+"/"+videoClass+"/"+fileInfo.baseName()+"/";
    QString imgFile = classPath+"image";
    QString xFlowFile = classPath+"flow_x";
    QString yFlowFile = classPath+"flow_y";
    QString flowFile = classPath+"flow";

    QDir dir;
    dir.mkpath(classPath);

    VideoCapture capture(videoFileName.toStdString());
    if(!capture.isOpened()) {
        printf("Could not initialize capturing..\n");
        return -1;
    }

    int frame_num = 0;
    Mat image, prev_image, prev_grey, grey, frame;
    //Mat flow_x, flow_y;
    GpuMat frame_0, frame_1; //, flow_u, flow_v;

    setDevice(gpu_id);
//    int devs = cv::cuda::getCudaEnabledDeviceCount();
//    std::cout << "devs: " << devs << std::endl;

    //FarnebackOpticalFlow alg_farn;
//    OpticalFlowDual_TVL1_GPU alg_tvl1;
//    BroxOpticalFlow alg_brox(0.197f, 50.0f, 0.8f, 10, 77, 10);

    // OpenCV 3
    Ptr<cuda::FarnebackOpticalFlow> alg_farn = cuda::FarnebackOpticalFlow::create();
    Ptr<cuda::OpticalFlowDual_TVL1> alg_tvl1 = cuda::OpticalFlowDual_TVL1::create();
    Ptr<cuda::BroxOpticalFlow> alg_brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);

    GpuMat d_flow(frame_0.size(), CV_32FC2);

    bool processed = false;
    while(true) {
        capture >> frame;
        if(frame.empty())
            break;

        if(frame_num == 0) {
            image.create(frame.size(), CV_8UC3);
            grey.create(frame.size(), CV_8UC1);
            prev_image.create(frame.size(), CV_8UC3);
            prev_grey.create(frame.size(), CV_8UC1);

            frame.copyTo(prev_image);
            cvtColor(prev_image, prev_grey, CV_BGR2GRAY);

            frame_num++;

            int step_t = step;
            while (step_t > 1){
                capture >> frame;
                step_t--;
            }
            continue;
        }


        // Check whether the frame already exists
        char tmp[20];
        sprintf(tmp,"_%04d.jpg",int(frame_num));

        QString xFlowFileFull = xFlowFile + QString(tmp);
        QString yFlowFileFull = yFlowFile + QString(tmp);
        QString imgFileFull = imgFile + QString(tmp);
        QString flowFileFull = flowFile + QString(tmp);

//        qDebug() << xFlowFileFull;
//        qDebug() << yFlowFileFull;
//        qDebug() << imgFileFull;
        if (overwrite ||
                !QFileInfo( flowFileFull).exists() ||
                !QFileInfo(xFlowFileFull).exists() ||
                !QFileInfo(yFlowFileFull).exists() ||
                !QFileInfo(imgFileFull).exists()){

            processed = true;
            frame.copyTo(image);
            cvtColor(image, grey, CV_BGR2GRAY);
    //        imshow("main", grey);
    //        waitKey(1);
    //        continue;

            frame_0.upload(prev_grey);
            frame_1.upload(grey);

    //        GpuMat frame_0(prev_grey);
    //        GpuMat frame_1(grey);

    //        GpuMat frame_0;
    //        GpuMat frame_1;
    //        frame_0_r.convertTo(frame_0, CV_32F, 1.0 / 255.0);
    //        frame_1_r.convertTo(frame_1, CV_32F, 1.0 / 255.0);


            // GPU optical flow
            switch(type){
            case 0:
    //            alg_farn->calc(frame_0,frame_1, flow_u,flow_v);
                alg_farn->calc(frame_0,frame_1, d_flow);
                break;
            case 1:
                alg_tvl1->calc(frame_0,frame_1, d_flow);
                break;
            case 2:
                GpuMat d_frame0f, d_frame1f;
                frame_0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
                frame_1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);
                alg_brox->calc(d_frame0f, d_frame1f, d_flow);
                break;
            }

            //printf("Frame: %d\n", frame_num);
            //showFlow("flow", d_flow);  waitKey(1);


    //		flow_u.download(flow_x);
    //		flow_v.download(flow_y);
            // Opencv 3.0
            GpuMat planes[2];
            cuda::split(d_flow, planes);
            Mat flow_x(planes[0]);
            Mat flow_y(planes[1]);

            // Output optical flow
            Mat imgX(flow_x.size(),CV_8UC1);
            Mat imgY(flow_y.size(),CV_8UC1);
            convertFlowToImage(flow_x, flow_y, imgX, imgY, -bound, bound);

            if (1){
                // Write out combined flow filed as RGB img
                int scale = 16;

//                cv::Mat mag = cv::Mat(xoff.rows, xoff.cols, CV_8UC1, cv::Scalar(128));
//                magf.convertTo(mag, CV_8UC1, 127.0, 128);
//                cv::minMaxLoc(mag, &minval, &maxval);

                cv::Mat magf;
                cv::magnitude(flow_x, flow_y, magf);

                magf = magf*scale+128;

                cv::Mat flow_x_s = flow_x*scale+128;
                cv::Mat flow_y_s = flow_y*scale+128;

                cv::Mat mag, xof, yof;
                magf.convertTo(mag, CV_8UC1, 1, 0);
                flow_x_s.convertTo(xof, CV_8UC1, 1, 0);
                flow_y_s.convertTo(yof, CV_8UC1, 1, 0);

                std::vector<cv::Mat> channels;
                channels.push_back(mag);
                channels.push_back(yof);
                channels.push_back(xof);

                cv::Mat out;
                cv::merge(channels, out);
                cv::imwrite(flowFileFull.toStdString(), out);
                //std::cout << flowFileFull.toStdString().c_str() << std::endl << std::flush;
            }


            if (1){

                Mat imgX_, imgY_, image_;
                resize(imgX,imgX_,cv::Size(340,256));
                resize(imgY,imgY_,cv::Size(340,256));
                resize(image, image_,cv::Size(340,256));

                imwrite(xFlowFile.toStdString() + tmp, imgX_);
                imwrite(yFlowFile.toStdString() + tmp, imgY_);
                imwrite(imgFile.toStdString() + tmp, image_);
            }
        }else{
           // std::cout << "[" << gpu_id << "] "<< "Already exist (rgb, xflow, yflow): " << imgFileFull.toStdString() <<  std::endl << std::flush;
        }

        std::swap(prev_grey, grey);
        std::swap(prev_image, image);
        frame_num = frame_num + 1;

        int step_t = step;
        while (step_t > 1){
            capture >> frame;
            step_t--;
        }
    }

    if (!processed)
        return -2;

    return frame_num;
}



void ImageProcessor::run()
{
//    Qt::HANDLE thandle = QThread::currentThreadId();
//    qDebug("Thread id inside run %X", thandle);

    frames = 0;
    //static int value=0; //If this is not static, then it is reset to 0 every time this function is called.
    isRunning = 1;
    while(true)  {

        // If there are no jobs check later
/*
        mutex->lock();
        if (!jobs || jobs->size() == 0){
            mutex->unlock();
            if (isRunning == 0)
                break;
            msleep(5);
            continue;
           // break;
        }

        // Get next job
        JobItem *job = jobs->first();
        jobs->removeFirst();
        int left = jobs->size();
        mutex->unlock();
*/
        JobItem job;
        jobQueue->popWait(job);
        int left = jobQueue->size();

        // And start work on it

        QString videoFileName = job.filename;
        QString videoClass = "";

        if (parseVideoClass){
            QStringList toks = videoFileName.split("_");
            if (toks.size()<4)
                continue;
            videoClass = toks[1];
        }
        std::cout << "[" << mpi_id << "/" << gpu_id << "  " << frames  << "-" << left << "] " << videoFileName.toStdString() << std::endl << std::flush;
        int videoFrames = process(videoClass, videoFileName);
        if (videoFrames == -2){
        }
        frames++;
    }

    std::cout << "[" << gpu_id << "] " << "Done Processed videos : " << frames;
}
