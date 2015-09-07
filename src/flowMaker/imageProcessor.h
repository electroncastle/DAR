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
        //ImageProcessor::jobs = jobs;
        //ImageProcessor::mutex = mutex;
        ImageProcessor::jobQueue = jobs;

    }


    int process(QString videoClass, QString videoFileName);

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

};


#endif
