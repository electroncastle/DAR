#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

// Without the std namespace here the opencv gpu module will complain about
// non existing vector<>
using namespace std;
//#include "opencv2/gpu/gpu.hpp"

#include <stdio.h>
#include <iostream>
#include <sys/time.h>
using namespace cv;
//using namespace cv::gpu;




#include <QCoreApplication>
#include <QtAlgorithms>
#include <QDirIterator>
#include <QDebug>
#include <QDir>
#include <extractFlow.h>
#include <colorcode.h>

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


Mat img_resize(Mat &src, cv::Size new_size, int resize_type)
{
    int cw = src.cols;
    int ch = src.rows;

    Mat result;
    if (resize_type == 2){
        Mat tmp;

        // Resize first to fit to the smallest size of the new bounding box
        // while keeping original aspect ratio
        // and then crop to the exact new width height
        float aspect = (float)cw/ch;

        int hd = new_size.height - ch;
        int wd = new_size.width - cw;

        if (hd*aspect > wd)
            cv::resize(src, tmp, cv::Size(cw+aspect*hd, new_size.height));
        else
            cv::resize(src, tmp, cv::Size(new_size.width, ch+wd/aspect));

        int x = (tmp.cols-new_size.width) / 2;
        int y = (tmp.rows-new_size.height) / 2;
        cv::Rect crop_rect(x, y, new_size.width, new_size.height);
        result = tmp(crop_rect);
//        cv::imshow("src", src);
//        cv::imshow("tmp", result);
//        cv::waitKey(0);
    }


    return result;
}

//int ImageProcessor::processSimple(QString img1, QString img2)
//{

//}


void ImageProcessor::initFlow()
{
    alg_farn = cuda::FarnebackOpticalFlow::create();
    alg_tvl1 = cuda::OpticalFlowDual_TVL1::create(
                0.25, // double tau =0.25

                /**
                 * Weight parameter for the data term, attachment parameter.
                 * This is the most relevant parameter, which determines the smoothness of the output.
                 * The smaller this parameter is, the smoother the solutions we obtain.
                 * It depends on the range of motions of the images, so its value should be adapted to each image sequence.
                 */
                0.15, // double lambda = 0.15

                /**
                 * parameter used for motion estimation. It adds a variable allowing for illumination variations
                 * Set this parameter to 1. if you have varying illumination.
                 * See: Chambolle et al, A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging
                 * Journal of Mathematical imaging and vision, may 2011 Vol 40 issue 1, pp 120-145
                 */
                0.3, // double theta = 0.3

                /**
                 * Number of scales used to create the pyramid of images.
                 */
                10, // int nscales = 5

                /**
                 * Number of warpings per scale.
                 * Represents the number of times that I1(x+u0) and grad( I1(x+u0) ) are computed per scale.
                 * This is a parameter that assures the stability of the method.
                 * It also affects the running time, so it is a compromise between speed and accuracy.
                 */
                10, // int warps = 5

                /**
                 * Stopping criterion threshold used in the numerical scheme, which is a trade-off between precision and running time.
                 * A small value will yield more accurate solutions at the expense of a slower convergence.
                 */
                0.01, //double epsilon = 0.01
                300, // int iterations = 300
                0.8, // double scaleStep =0.8


                /**
                 * Weight parameter for (u - v)^2, tightness parameter.
                 * It serves as a link between the attachment and the regularization terms.
                 * In theory, it should have a small value in order to maintain both parts in correspondence.
                 * The method is stable for a large range of values of this parameter.
                 */
                0.000, // double gamma =
                false); //bool useInitialFlow =

    alg_brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);

    //GpuMat d_flow(frame_0.size(), CV_32FC2);
//    d_flow = GpuMat(frame_0.size(), CV_32FC2);
}

void ImageProcessor::estimateFlow(QString img1_file, QString img2_file, QString colorFlow_file, int of_type, cv::Size new_size)
{
    cv::Mat img1 = cv::imread(img1_file.toStdString(), cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(img2_file.toStdString(), cv::IMREAD_GRAYSCALE);

    Mat flow_x;
    Mat flow_y;
    int flow_type = 1;
    estimateFlow(img1, img2, flow_x, flow_y, flow_type);

    double  bound = 30.0;
    bound = 20.0;
    cv::Mat rgbFlow = motionToColor(flow_x, flow_y, bound);


//    cv::Size new_size(224, 224);
    int resize_type = 2;
    if (new_size.width > 0){
        rgbFlow = img_resize(rgbFlow, new_size, resize_type);
    }


    vector<int> p;
    p.push_back(CV_IMWRITE_JPEG_QUALITY);
    p.push_back(99); // compression factor
    imwrite(colorFlow_file.toStdString(), rgbFlow, p);

}


void ImageProcessor::estimateFlow(cv::Mat &img1, cv::Mat &img2, cv::Mat &flow_x, cv::Mat &flow_y, int of_type)
{
#define DO_FLOW 1
#ifdef DO_FLOW

            frame_0.upload(img1);
            frame_1.upload(img2);

            d_flow = GpuMat(frame_0.size(), CV_32FC2);

    //        GpuMat frame_0(prev_grey);
    //        GpuMat frame_1(grey);

    //        GpuMat frame_0;
    //        GpuMat frame_1;
    //        frame_0_r.convertTo(frame_0, CV_32F, 1.0 / 255.0);
    //        frame_1_r.convertTo(frame_1, CV_32F, 1.0 / 255.0);


            // GPU optical flow
            switch(of_type){
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
            flow_x = cv::Mat(planes[0]);
            flow_y = cv::Mat(planes[1]);
#else
            Mat flow_x = cv::Mat(240, 320, CV_32FC1, cv::Scalar(0.5));
            Mat flow_y  = cv::Mat(240, 320, CV_32FC1, cv::Scalar(0.5));
#endif

}

void ImageProcessor::init()
{
    initFlow();
}

int ImageProcessor::process(QString videoClass, QString videoFileName)
{
    vector<int> p;
    p.push_back(CV_IMWRITE_JPEG_QUALITY);
    p.push_back(99); // compression factor

    bool overwrite = false;
    int step = 1;
    int type = 1;
    int bound = 20;

    int new_width = 224;//320;
    int new_height = 224; //256;
    cv::Size new_size(new_width, new_height);

    // 0 - stretch
    // 1 - fit
    // 2 - resize to the target size with aspect ratio 1:1 and then crop outside region
    int resize_type = 2;
    //bool preserve_aspect_ratio = true;


    //--------------------------

    QFileInfo fileInfo(videoFileName);
    QString classPath = path+"/"+videoClass+"/"+fileInfo.baseName()+"/";
    QString imgFile = classPath+"cimage";
    QString xFlowFile = classPath+"flow_x";
    QString yFlowFile = classPath+"flow_y";
    QString flowFile = classPath+"flow";

    // Don't produce the x/y flows individually
    xFlowFile = "";
    yFlowFile = "";


    QDir dir;
    dir.mkpath(classPath);

    VideoCapture capture(videoFileName.toStdString());
    if(!capture.isOpened()) {
        printf("Could not initialize capturing..\n");
        return -1;
    }

    int frame_num = 0;
    Mat image, prev_image, prev_grey, grey, frame;
    GpuMat frame_0, frame_1;

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

    //GpuMat d_flow(frame_0.size(), CV_32FC2);


    bool processed = false;
    struct timeval now, last;
    gettimeofday(&last, NULL);
    int fpsFrames = 0;
    while(true) {
        capture >> frame;
        if(frame.empty())
            break;


        // start timer
        gettimeofday(&now, NULL);
        fpsFrames++;

        // Compute and print the elapsed time in millisec
        double elapsedTime;
        elapsedTime = (now.tv_sec - last.tv_sec) * 1000.0;      // sec to ms
        elapsedTime += (now.tv_usec - last.tv_usec) / 1000.0;   // us to ms

        if (elapsedTime >= 1000.0){
            float fps = 1000.0*fpsFrames/elapsedTime;
            std::cout << "[" << this->mpi_id << "-" <<frame_num << "  fps: " << fps << std::endl << std::flush;

            fpsFrames=0;
            last = now;
        }




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
        QString imgFileFull = imgFile != "" ? imgFile + QString(tmp) : "";

        // The OF is calculated prev_frame -> current_frame
        // The OF image represents motion of pixels from prev_frame -> current_frame
        // this the OF image will have index of hte prev_frame
        sprintf(tmp,"_%04d.jpg",int(frame_num-1));
        QString xFlowFileFull = xFlowFile != "" ? xFlowFile + QString(tmp) : "";
        QString yFlowFileFull = yFlowFile != "" ? yFlowFile + QString(tmp) : "";
        QString flowFileFull = flowFile != "" ? flowFile + QString(tmp) : "";

        //printf("Decoded frame: %d\n", frame_num);

        // If the image/flow filename is specified and the file doesn't
        // exist on the filesystem run the optical flow estimation for
        // all images/flows
        if (overwrite ||
                (flowFileFull != "" && !QFileInfo(flowFileFull).exists()) ||
                (xFlowFileFull != "" && !QFileInfo(xFlowFileFull).exists()) ||
                (yFlowFileFull != "" && !QFileInfo(yFlowFileFull).exists()) ||
                (imgFileFull != "" && !QFileInfo(imgFileFull).exists()) )
        {

            processed = true;
            frame.copyTo(image);
            cvtColor(image, grey, CV_BGR2GRAY);
    //        imshow("main", grey);
    //        waitKey(1);
    //        continue;

            Mat flow_x;
            Mat flow_y;
            estimateFlow(prev_grey, grey, flow_x, flow_y, type);
/*
#define DO_FLOW 1
#ifdef DO_FLOW

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
#else
            Mat flow_x = cv::Mat(240, 320, CV_32FC1, cv::Scalar(0.5));
            Mat flow_y  = cv::Mat(240, 320, CV_32FC1, cv::Scalar(0.5));
#endif
*/

//            qDebug() << xFlowFileFull;
//            qDebug() << yFlowFileFull;
//            qDebug() << imgFileFull;
//            qDebug() << flowFileFull;

            ImageItem ii;
            ii.set(flow_x, flow_y, image, bound,
                   imgFileFull,
                   xFlowFileFull, yFlowFileFull,
                   flowFileFull,
                   new_size, resize_type);
            imageQueue->push(ii);

#ifdef XXX
if(0){
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

                if (threaded_imgwrite){
                    ImageItem ii;
                    ii.filename = flowFileFull;
                    ii.new_size = cv::Size(0,0);
                    ii.resize_type = 0;
                    ii.image = out;
                    imageQueue->push(ii);
                }else{
                    cv::imwrite(flowFileFull.toStdString(), out, p);
                }


                //std::cout << flowFileFull.toStdString().c_str() << std::endl << std::flush;
            }


            if (1){
                Mat imgX_, imgY_, image_;
//                resize(imgX,imgX_,cv::Size(340,256));
//                resize(imgY,imgY_,cv::Size(340,256));
//                resize(image, image_,cv::Size(340,256));

                if (threaded_imgwrite){
                    ImageItem ii;
                    ii.set(imgX, xFlowFile + tmp, new_size, resize_type); imageQueue->push(ii);
                    ii.set(imgY, yFlowFile + tmp, new_size, resize_type); imageQueue->push(ii);
                    ii.set(image, imgFile + tmp, new_size, resize_type); imageQueue->push(ii);
                }else{
                    imgX_ = img_resize(imgX, new_size, resize_type);
                    imgY_ = img_resize(imgY, new_size, resize_type);
                    image_ = img_resize(image, new_size, resize_type);

                    imwrite(xFlowFile.toStdString() + tmp, imgX_, p);
                    imwrite(yFlowFile.toStdString() + tmp, imgY_, p);
                    imwrite(imgFile.toStdString() + tmp, image_, p);
                }
            }
            //std::cout << "OF done: " << imgFile.toStdString().c_str() << std::endl << std::flush;
}
#endif

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

    std::cout << "VIDEO done: " << imgFile.toStdString().c_str() << std::endl << std::flush;

    if (!processed)
        return -2;

    return frame_num;
}



void ImageProcessor::run()
{
//    Qt::HANDLE thandle = QThread::currentThreadId();
//    qDebug("Thread id inside run %X", thandle);

    threaded_imgwrite = true;
    imgWriter = new ImageWriter();
    imageQueue = new ConcurrentQueue<ImageItem>();
    imgWriter->setImageQueue(imageQueue);
    imgWriter->start();

    frames = 0;
    //static int value=0; //If this is not static, then it is reset to 0 every time this function is called.
    isRunning = 1;
    while(true)  {

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
//        std::cout << "[" << mpi_id << "/" << gpu_id << "  " << frames  << "-" << left << "] " << videoFileName.toStdString() << std::endl << std::flush;
        std::cout << "[mpi=" << mpi_id << " gpu=" << gpu_id << "  frames_done=" << frames   << "] " << videoFileName.toStdString() << std::endl << std::flush;
        int videoFrames = process(videoClass, videoFileName);
        if (videoFrames == -2){
        }
        std::cout << "[" << gpu_id << "] " << videoFileName.toStdString() << " Done. Frames: " << frames << std::endl << std::flush;

        frames++;
    }

    std::cout << "[" << gpu_id << "] " << "Done Processed videos : " << frames << std::endl << std::flush;
}





cv::Mat flowToRGB(cv::Mat &flow_x, cv::Mat &flow_y )
{
        // Write out combined flow filled as RGB img
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

//                if (threaded_imgwrite){
//                    ImageItem ii;
//                    ii.filename = flowFileFull;
//                    ii.new_size = cv::Size(0,0);
//                    ii.resize_type = 0;
//                    ii.image = out;
//                    imageQueue->push(ii);
//                }else{
//                    cv::imwrite(flowFileFull.toStdString(), out, p);
//                }


        //std::cout << flowFileFull.toStdString().c_str() << std::endl << std::flush;
    return out;
}

//**************************************************************************************
//      ImageWriter
//**************************************************************************************
void ImageWriter::run()
{
    vector<int> p;
    p.push_back(CV_IMWRITE_JPEG_QUALITY);
    p.push_back(95); // compression factor

    isRunning = 1;

    while(isRunning)  {
        ImageItem img;
        imageQueue->popWait(img);
       // int left = imageQueue->size();

        // Output optical x,y flow images
        if (img.filename_flowx != ""){
            Mat imgX(img.flow_x.size(), CV_8UC1);
            Mat imgY(img.flow_y.size(), CV_8UC1);

            convertFlowToImage(img.flow_x, img.flow_y, imgX, imgY, -img.bound, img.bound);

            Mat imgX_, imgY_;
            imgX_ = img_resize(imgX, img.new_size, img.resize_type);
            imgY_ = img_resize(imgY, img.new_size, img.resize_type);

            imwrite(img.filename_flowx.toStdString(), imgX_, p);
            imwrite(img.filename_flowy.toStdString(), imgY_, p);
        }

        // Output RGB flow image
        if (img.filename_flow != ""){
            //cv::Mat rgbFlow = flowToRGB(img.flow_x, img.flow_y);
            cv::Mat rgbFlow = motionToColor(img.flow_x, img.flow_y, 20.0);
            cv::Mat rgbFlow_ = img_resize(rgbFlow, img.new_size, img.resize_type);
            imwrite(img.filename_flow.toStdString(), rgbFlow_, p);
        }


        // Write out RGB image
        if (img.filename != ""){
            cv::Mat image_ = img_resize(img.image, img.new_size, img.resize_type);
            imwrite(img.filename.toStdString(), image_, p);
        }

    }

}

void ImageWriter::finish()
{
    isRunning = 0;
}
