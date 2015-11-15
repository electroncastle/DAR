#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H
#include <stdio.h>
#include <iostream>
#include <queue>

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

// Without the std namespace here the opencv gpu module will complain about
// non existing vector<>
using namespace std;
//#include "opencv2/gpu/gpu.hpp"

using namespace cv;
//using namespace cv::gpu;

// Opoencv 3
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"


#include <QtCore>
#include <QThread>
#include <QList>


void drawOpticalFlow(const Mat_<float>& flowx, const Mat_<float>& flowy, Mat& dst, float maxmotion = -1);
void showFlow(const char* name, const cv::cuda::GpuMat& d_flow);


template<typename Data>
class ConcurrentQueue
{
private:
    std::queue<Data> the_queue;
    QMutex the_mutex;
    QWaitCondition write_condition;
    QWaitCondition read_condition;
    bool closed;

public:

    int size()
    {
        QMutexLocker locker(&the_mutex);
        return the_queue.size();
    }

    void setClosed(bool state)
    {
        QMutexLocker locker(&the_mutex);
        closed = state;
    }

    bool getClosed()
    {
        QMutexLocker locker(&the_mutex);
        return closed;
    }

    void push(Data const& data)
    {
        QMutexLocker locker(&the_mutex);
        the_queue.push(data);
        read_condition.wakeOne();
    }

    bool empty()
    {
        QMutexLocker locker(&the_mutex);
        return the_queue.empty();
    }

//    bool try_pop(Data& popped_value)
//    {
//        QMutexLocker locker(&the_mutex);
//        if(the_queue.empty())
//        {
//            return false;
//        }
//        popped_value = the_queue.front();
//        the_queue.pop();
//        return true;
//    }

    void popWait(Data& popped_value)
    {
        QMutexLocker locker(&the_mutex);
        while(the_queue.empty()) {
            read_condition.wait(&the_mutex);
        }
        popped_value = the_queue.front();
        the_queue.pop();
        write_condition.wakeOne();
    }

    //created to allow for a limited queue size
    void pushWait(Data const& data, const int max_size)
    {
        QMutexLocker locker(&the_mutex);
        while(the_queue.size() >= max_size)  {
            write_condition.wait(&the_mutex);
        }
        the_queue.push(data);
        read_condition.wakeOne();
    }


};

class JobItem//: public QObject
{
//  Q_OBJECT

public:
    QString filename;
    QString className;
    int label;
};


class ImageItem
{
public:
    void set(cv::Mat image, QString filename, cv::Size new_size=cv::Size(0,0), int resize_type=0){
        ImageItem::image = image;
        ImageItem::filename = filename;
        ImageItem::new_size  = new_size;
        ImageItem::resize_type = resize_type;
    }

    void set(cv::Mat flow_x, cv::Mat flow_y, cv::Mat image, int bound,
             QString filename,
             QString filename_flowx,
             QString filename_flowy,
             QString filename_flow,
             cv::Size new_size=cv::Size(0,0), int resize_type=0){

        ImageItem::image = image;
        ImageItem::filename = filename;
        ImageItem::filename_flowx = filename_flowx;
        ImageItem::filename_flowy = filename_flowy;
        ImageItem::filename_flow = filename_flow;
        ImageItem::new_size  = new_size;
        ImageItem::resize_type = resize_type;

        ImageItem::flow_x = flow_x;
        ImageItem::flow_y = flow_y;
        ImageItem::bound = bound;
    }

    cv::Size new_size;
    int resize_type;
    cv::Mat image;
    QString filename;
    QString filename_flowx;
    QString filename_flowy;
    QString filename_flow;

    // Source flow from the estimator
    cv::Mat flow_x;
    cv::Mat flow_y;
    int bound;
};

class ImageWriter: public QThread
{
    Q_OBJECT

 public:

    void setImageQueue(ConcurrentQueue<ImageItem> *imageQueue){
        ImageWriter::imageQueue = imageQueue;
    }

protected:
   virtual void run();

public slots:
    void finish();

private:
    ConcurrentQueue<ImageItem> *imageQueue;
    bool isRunning;
};


class ImageProcessor : public QThread
{
   Q_OBJECT

public:
    ImageProcessor(){
        ImageProcessor::parseVideoClass = false;
    }

    void setGPUDevice(int gpu_id){
        ImageProcessor::gpu_id = gpu_id;
    }

    void setMPIId(int mpi_id){
        ImageProcessor::mpi_id = mpi_id;
    }

    void setPath(QString path){
        ImageProcessor::path = path;
    }

    void setJobList(ConcurrentQueue<JobItem> *jobs){
        ImageProcessor::jobQueue = jobs;
    }

    void init();
    int process(QString videoClass, QString videoFileName);

    void estimateFlow(QString img1_file, QString img2_file, QString colorFlow_file, int of_type, Size new_size=cv::Size(0,0) );
    void estimateFlow(Mat &img1, Mat &img2, cv::Mat &flow_x, cv::Mat &flow_y, int of_type);
    void initFlow();

public slots:
    void finish();

protected:
   virtual void run();

signals:
   void signalValueUpdated(QString);

private:

   bool isRunning;
    int gpu_id;
    int mpi_id;
  //  QList<JobItem*> *jobs;
  //  QMutex *mutex;
    ConcurrentQueue<JobItem> *jobQueue;
    QString path;
    int frames;
    bool parseVideoClass;

    bool threaded_imgwrite;
    ConcurrentQueue<ImageItem> *imageQueue;
    ImageWriter *imgWriter;


    Ptr<cuda::FarnebackOpticalFlow> alg_farn;
    Ptr<cuda::OpticalFlowDual_TVL1> alg_tvl1;
    Ptr<cuda::BroxOpticalFlow> alg_brox;
    cv::cuda::GpuMat frame_0;
    cv::cuda::GpuMat frame_1;
    cv::cuda::GpuMat d_flow;
};


#endif
