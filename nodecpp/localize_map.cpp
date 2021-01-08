//
// Created by qzj on 2020/12/27.
//
#include "localize_map.h"
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/range_image/range_image.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include "ros/ros.h"
#include <string>
#include <vector>
#include "sensor_msgs/PointCloud2.h"
#include "selfDefine.h"
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>

using namespace std;
using namespace Eigen;

class LocalizeMap {
private:

    ros::NodeHandle nh;

    pcl::PointCloud<PointType>::Ptr laserCloudIn;
    pcl::PointCloud<PointType>::Ptr transformedCloudIn;
    pcl::PointCloud<PointType>::Ptr priorMapCloud;
    pcl::PointCloud<PointType>::Ptr nearSubMap;
    pcl::PointCloud<PointType>::Ptr cloudSourceAftReg;
    PointType nanPoint; // fill in fullCloud at each iteration

    ros::Publisher pubRawScan;
    ros::Publisher pubNearSubMap;
    ros::Publisher pubGlobalMap;
    ros::Subscriber subLaserCloud;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeNearSubMap;

    string path_to_map;
    
    Eigen::Matrix4f T_cur;
    Eigen::Matrix4f T_prev;
    Eigen::Matrix4f T_scan_map;

    std_msgs::Header cloudHeader;

    //NDT Parameters
    double _ndtEpsilon; //minimum transformation difference for termination condition.
    double _ndtStepSize; //maximum step size for More-Thuente line search
    double _ndtFitnessThreshold; // Fitness means the average distance between source points and target points whose distance less than some certain distance
    float _ndtGridResolution; //Resolution of NDT grid structure (VoxelGridCovariance)
    int _ndtMaxIterations; //max number of registration iterations

public:
    LocalizeMap() :
            _ndtEpsilon(0.01), _ndtStepSize(0.1), _ndtFitnessThreshold(0.02),
            _ndtGridResolution(1.0), _ndtMaxIterations(10),
            nh("~") {

        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 1, &LocalizeMap::cloudHandler, this);
        pubRawScan = nh.advertise<sensor_msgs::PointCloud2>("/raw_scan", 1);
        pubNearSubMap = nh.advertise<sensor_msgs::PointCloud2>("/nearSubMap", 1);
        pubGlobalMap = nh.advertise<sensor_msgs::PointCloud2>("/global_map", 1);

        nh.param<std::string>("path_to_map", path_to_map, "");
        if(path_to_map.compare("")==0)
            ROS_ERROR("failed to load map!");

        allocateMemory();
        resetParameters();

        pcl::PointCloud<PointType>::Ptr priorMapCloudRaw(new pcl::PointCloud<PointType>());
        if (pcl::io::loadPCDFile<PointType> (path_to_map, *priorMapCloudRaw) == -1)
            ROS_ERROR ("Couldn't read file %s \n", path_to_map.c_str());
        else
            ROS_INFO("Load map with %d points.", priorMapCloudRaw->points.size());
        Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
        trans(0,0)=0;trans(1,1)=0;trans(2,2)=0;
        trans(0,2)=1;trans(1,0)=1;trans(2,1)=1;
        pcl::transformPointCloud(*priorMapCloudRaw, *priorMapCloud, trans);

        kdtreeNearSubMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeNearSubMap->setInputCloud(priorMapCloud);

        ROS_INFO("LocalizeMap initialize finish");
    }

    void allocateMemory() {

        laserCloudIn.reset(new pcl::PointCloud<PointType>());
        transformedCloudIn.reset(new pcl::PointCloud<PointType>());
        priorMapCloud.reset(new pcl::PointCloud<PointType>());
        nearSubMap.reset(new pcl::PointCloud<PointType>());
        cloudSourceAftReg.reset(new pcl::PointCloud<PointType>());

        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;
        
        T_cur = Eigen::Matrix4f::Identity();
        T_prev = T_cur;
        T_scan_map = Eigen::Matrix4f::Identity();

    }

    void resetParameters(){

        laserCloudIn->clear();
        transformedCloudIn->clear();
        priorMapCloud->clear();
        nearSubMap->clear();
        cloudSourceAftReg->clear();

        T_cur = Eigen::Matrix4f::Identity();
        T_prev = T_cur;
        T_scan_map = Eigen::Matrix4f::Identity();
    }

    ~LocalizeMap() {}

    void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        cloudHeader = laserCloudMsg->header;
        // cloudHeader.stamp = ros::Time::now(); // Ouster lidar users may need to uncomment this line
        pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);
        // Remove Nan points
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        // 1. Convert ros message to pcl point cloud
        copyPointCloud(laserCloudMsg);
        
        findNearSubMap();

        zoom();

        ndtRegistration();

        update();

        publishCloud();

    }

    void update()
    {
        T_prev = T_cur;
        pcl::transformPointCloud(*laserCloudIn, *transformedCloudIn, T_cur);
    }

    void ndtRegistration()
    {
        ros::Time beginTime = ros::Time::now();

        //NDT initialization
        pcl::NormalDistributionsTransform<PointType , PointType> ndt_solver;
        ndt_solver.setTransformationEpsilon(_ndtEpsilon);
        ndt_solver.setStepSize(_ndtStepSize);
        ndt_solver.setResolution(_ndtGridResolution);
        ndt_solver.setMaximumIterations(_ndtMaxIterations);
        //Load clouds
        ndt_solver.setInputSource(laserCloudIn);
        ndt_solver.setInputTarget(nearSubMap);

        //Align and get results
        ndt_solver.align(*cloudSourceAftReg, T_cur);
        bool hasConverged = ndt_solver.hasConverged();
        int FinalNumIteration = ndt_solver.getFinalNumIteration();
        double Score = ndt_solver.getFitnessScore(0.5);

        if (!hasConverged){
            ROS_ERROR("NDT has not converged!");
        }
        else{

            T_scan_map = ndt_solver.getFinalTransformation();
            T_cur = T_scan_map;

            ros::Time endTime = ros::Time::now();
            ROS_INFO("Time consumption of NDT matching: %f seconds. Iteration %d times. Score: %f."
                     ,(endTime - beginTime).toSec(), FinalNumIteration,Score);
        }
    }

    void zoom()
    {
        pcl::PassThrough<PointType> passThroughFilter_x;     //创建滤波器对象
        passThroughFilter_x.setFilterFieldName("x");
        passThroughFilter_x.setFilterLimits(-nearSubMapSearchRadius, nearSubMapSearchRadius);
        passThroughFilter_x.setFilterLimitsNegative(false);      //保留
        passThroughFilter_x.setInputCloud(laserCloudIn);                //设置待滤波的点云
        passThroughFilter_x.filter(*laserCloudIn);               //滤波并存储

        pcl::PassThrough<PointType> passThroughFilter_y;     //创建滤波器对象
        passThroughFilter_y.setFilterFieldName("y");
        passThroughFilter_y.setFilterLimits(-nearSubMapSearchRadius, nearSubMapSearchRadius);
        passThroughFilter_y.setFilterLimitsNegative(false);      //保留
        passThroughFilter_y.setInputCloud(laserCloudIn);                //设置待滤波的点云
        passThroughFilter_y.filter(*laserCloudIn);               //滤波并存储
    }

    void findNearSubMap() {

        PointType currentRobotPosPoint;
        currentRobotPosPoint.x = T_cur(0,3);
        currentRobotPosPoint.y = T_cur(1,3);
        currentRobotPosPoint.z = T_cur(2,3);

        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        kdtreeNearSubMap->radiusSearch(currentRobotPosPoint, nearSubMapSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);

        nearSubMap->clear();
        for (int i = 0; i < pointSearchIndGlobalMap.size(); ++i)
            nearSubMap->points.push_back(priorMapCloud->points[pointSearchIndGlobalMap[i]]);


    }

    void publishCloud() {
        static int frequency = 0;
        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*transformedCloudIn, output);   // 转换成ROS下的数据类型 最终通过topic发布
        output.header.stamp = ros::Time::now();
        output.header.frame_id = "slam";
        pubRawScan.publish(output);

        pcl::toROSMsg(*nearSubMap, output);   // 转换成ROS下的数据类型 最终通过topic发布
        output.header.stamp = ros::Time::now();
        output.header.frame_id = "slam";
        pubNearSubMap.publish(output);

        if(frequency++ % 10==0)
        {
            pcl::toROSMsg(*priorMapCloud, output);
            output.header.stamp = ros::Time::now();
            output.header.frame_id = "slam";
            pubGlobalMap.publish(output);
        }
    }

};

int main(int argc, char **argv) {

    ros::init(argc, argv, "localize_map");

    ROS_INFO("\033[1;32m---->\033[0m localize_map Started.");

    LocalizeMap localizeMap;

    ros::spin();

    return 0;
}




