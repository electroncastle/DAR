/*
 * Jiri Fajtl <ok1zjf@gmail.com>
 *
 * Loosely based on OpenCV CUDA examples and Limin Wang code dense_flow.
 *
 */


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

int myid, numprocs;


bool jobComp(JobItem* left, JobItem *right)
{
  return left->filename.compare(right->filename) < 0;
}

QMutex mutex;


QList<JobItem*> *jobsFromPath(QString path)
{
     QList<JobItem*> *jobs = new QList<JobItem*>();
     QDirIterator it(path, QDirIterator::Subdirectories);
     while (it.hasNext()) {
         QString filename =  it.next();
         if (!it.fileInfo().isFile()){
             continue;
         }

         QStringList toks = filename.split("_");
         if (toks.size()<4)
             continue;

         QString video_class = toks[1];

         //std::cout << filename.toStdString() << std::flush;
        // int frames = process(video_class, filename, rgbflow_path, 0);
        // std::cout << "  " << frames  << std::endl << std::flush;
         JobItem *job = new JobItem();
         job->filename = filename;
         job->className = video_class;
         jobs->push_back(job);
     }

     qSort(jobs->begin(), jobs->end(), jobComp);
     return jobs;
}

QList<JobItem*> *jobsFromFile(QString filename)
{
     QList<JobItem*> *jobs = new QList<JobItem*>();

     QFileInfo fi(filename);
     QFile file(filename);
     if(!file.open(QIODevice::ReadOnly)) {
         return NULL;
     }

     QTextStream in(&file);

     while(!in.atEnd()) {

         QString line = in.readLine();
         //QStringList fields = line.split(",");
         //model->appendRow(fields);

         JobItem *job = new JobItem();
         job->filename = fi.path()+"/"+line;
         job->className = "";
         jobs->push_back(job);
     }

     return jobs;
}

#ifdef USE_MPI
void run()
{
    bool makeRGBFlow = false;
    String data_root = "/home/jiri/Lake/HAR/datasets/UCF-101/";
    data_root = "/home/jiri/Lake/HAR/datasets/UCF-101/";

    // Source
    String video_path = data_root +"/UCF101/";
//    String train_split = data_root + "/ucfTrainTestlist/";
    QString rgbflow_path = QString::fromStdString(data_root+"/UCF101-rgbflow/");

    video_path = "/home/jiri/Lake/DAR/share/datasets/THUMOS2015/thumos15_validation/";
    rgbflow_path = "/home/jiri/Lake/DAR/share/datasets/THUMOS2015/thumos15_validation-rgbflow/";

    // output files
//    String rgb_img_path = data_root +"/rgb/";
//    String flow_img_path = data_root +"/flow/";
//    String labels_file = data_root + "labels.txt";
//    String train_file = data_root + "/train.txt";
//    String test_file = data_root + "/test.txt";

    if (myid == 0){

        //MASTER
//        QList<JobItem*> *jobs = jobsFromPath(QString::fromStdString(video_path));
//        QList<JobItem*> *jobs = jobsFromFile(QString::fromStdString(video_path+"/val.txt"));
        QList<JobItem*> *jobs = jobsFromFile(QString::fromStdString(video_path+"/annotated.txt"));


        int videoNum = 0;
        int devs=0;
        ImageProcessor *im[4];
        //QList<JobItem*> *clientJobs = new QList<JobItem*>();

        ConcurrentQueue<JobItem> *clientJobs = new ConcurrentQueue<JobItem>();

        if (numprocs == 1){
            // We are alone here - no MPI
            devs = cv::cuda::getCudaEnabledDeviceCount();
            std::cout << "** NO MPI" << std::endl;
            std::cout << "GPU CUDA devices: " << devs << std::endl;
            //devs = 1;

            for (int i=0; i<devs; i++){
                im[i] = new ImageProcessor();
                im[i]->setGPUDevice(i);
                im[i]->setMPIId(1);
                im[i]->setPath(rgbflow_path);
                im[i]->setJobList(clientJobs);
                im[i]->start();
            }
        }

        // Start the main process
        int maxClasses = jobs->size();// /2;
        // maxClasses = 6;
        for (int i=0; i<maxClasses; i++){

            JobItem *job = jobs->at(i);
            QString filename = job->filename;
            std::cout << "* Loading new job  " << filename.toStdString()  << std::endl << std::flush;

            // Filter
             if (0){
                QFileInfo fi(filename);
                if (fi.baseName() != "thumos15_video_validation_0001011"){
                    continue;
                }

    //            if (job->className == "BabyCrawling"){
    //                break;
    //            }

              }

//            if (makeRGBFlow){
//                makeRFBFlowImage(rgbflow_path, filename);
////                return;
//                continue;
//            }

            if (numprocs>1){

                qDebug() << "* Waitig for worker...";

                //MPI_Request request;
                MPI_Status status;
                int source = MPI_ANY_SOURCE;
                int val=0;
                MPI_Recv(&val, 1, MPI_INT, source, 100, MPI_COMM_WORLD, &status);

                char buffer[256];
                int count=256;
                int destination = status.MPI_SOURCE;
                qDebug()  << "* Serving item: " << i << "/" << maxClasses << " worker: " << destination << " video: " << filename;
                strcpy(buffer, filename.toLatin1().constData());
                MPI_Send(buffer, count, MPI_CHAR, destination, 200, MPI_COMM_WORLD);

            }else{
                // NO MPI
               // qDebug()  << "* " << myid << " Serving item: " << i << "/" << maxClasses <<  " video: " << filename;
                clientJobs->pushWait(*job, 5);
            }

            videoNum++;
        } // for (...)

        // If we are running without MPI wait for the workers
         if (numprocs==1){

             for (int i=0; i<devs; i++){
                 im[i]->finish();
             }

             for (int i=0; i<devs; i++){
                 im[i]->wait();
             }
         }

    }else{
        // WORKER
        //QList<JobItem*> *jobs = new QList<JobItem*>();
        ConcurrentQueue<JobItem> *clientJobs = new ConcurrentQueue<JobItem>();

        int devs = cv::cuda::getCudaEnabledDeviceCount();
        std::cout << "[" <<myid << "] Worker started" << std::endl;
        std::cout << "[" <<myid << "] SLAVE GPU CUDA devices: " << devs << std::endl;

        ImageProcessor *im[4];

        for (int i=0; i<devs; i++){
            im[i] = new ImageProcessor();
            im[i]->setGPUDevice(i);
            im[i]->setPath(rgbflow_path);
            im[i]->setJobList(clientJobs);
            im[i]->setMPIId(myid);
            im[i]->start();
        }

        QThread::sleep(myid);

        while (true){

            // Requesting new job
            qDebug() << "[" <<myid << "] Asking for job";
            int destination = 0;
            int val=0;
            MPI_Send(&val, 1, MPI_INT, destination, 100, MPI_COMM_WORLD);

            char buffer[256];
            int count=256;

            // Waiting for new job respose
            std::cout << "[" << myid << "] Waiting for new job respose" << std::endl << std::flush;

            MPI_Status status;
            int source = 0;
            MPI_Recv(buffer, count, MPI_CHAR, source, 200, MPI_COMM_WORLD, &status);

            QString filename = QString(buffer);
            qDebug() << "[" << myid << "] Got new job: " << filename;

            JobItem job;
            job.filename = filename;
            clientJobs->pushWait(job, 2);
        }
    }
}


#else

void run()
{
    String data_root = "/home/jiri/Lake/HAR/datasets/UCF-101/";

    // Source
    String video_path = data_root +"/UCF101/";
    String train_split = data_root + "/ucfTrainTestlist/";
    QString rgbflow_path = QString::fromStdString(data_root+"/UCF101-rgbflow/");

    // output files
    String rgb_img_path = data_root +"/rgb/";
    String flow_img_path = data_root +"/flow/";
    String labels_file = data_root + "labels.txt";
    String train_file = data_root + "/train.txt";
    String test_file = data_root + "/test.txt";

    QList<JobItem*> *jobs = new QList<JobItem*>();

    QDirIterator it(video_path.c_str(), QDirIterator::Subdirectories);
    while (it.hasNext()) {
        QString filename =  it.next();
        if (!it.fileInfo().isFile()){
            continue;
        }

        QStringList toks = filename.split("_");
        if (toks.size()<4)
            continue;

        QString video_class = toks[1];

        //std::cout << filename.toStdString() << std::flush;
       // int frames = process(video_class, filename, rgbflow_path, 0);
       // std::cout << "  " << frames  << std::endl << std::flush;

        JobItem *job = new JobItem();
        job->filename = filename;
        jobs->push_back(job);
//        if (jobs->size() == 3) break;
    }

    int devs = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "GPU CUDA devices: " << devs << std::endl;

    ImageProcessor *im[4];

    for (int i=0; i<devs; i++){
        im[i] = new ImageProcessor();
        im[i]->setGPUDevice(i);
        im[i]->setPath(rgbflow_path);
        im[i]->setJobList(jobs);
        im[i]->start();
    }

    for (int i=0; i<devs; i++){
        im[i]->wait();
    }

//    ImageProcessor im1;
//    im1.setGPUDevice(0);
//    im1.setPath(rgbflow_path);
//    im1.setJobList(jobs);

//    ImageProcessor im2;
//    im2.setGPUDevice(1);
//    im2.setPath(rgbflow_path);
//    im2.setJobList(jobs);

//    im1.start();
//    im2.start();

//    im1.wait();
//    im2.wait();
   // im1.sleep(10000000);
}
#endif

int main(int argc, char** argv)
{       
#ifdef USE_MPI
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
#endif

    int devs = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << myid << " =  GPU CUDA devices: " << devs << std::endl;

    QCoreApplication a(argc, argv);
    run();
    //return a.exec();

#ifdef USE_MPI
    MPI_Finalize();
#endif
}

