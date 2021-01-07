//
// Created by qzj on 2020/12/28.
//

#include "localize.h"
#include <yaml-cpp/yaml.h>

using namespace std;
using namespace Eigen;
#ifdef CUDA_FOUND
#include <ndt_gpu/NormalDistributionsTransform.h>
#endif

namespace lidar_localization {

    LocalizeMap::LocalizeMap() : local_map_ptr_(new CloudData::CLOUD()),
                                 current_scan_transformed_ptr_(new CloudData::CLOUD()),
                                 global_map_ptr_(new CloudData::CLOUD()), current_scan_ptr_(new CloudData::CLOUD()),
                                 nh("~") {

        InitParameters();

        InitFilterALL();

        InitGlobalMap();

        ROSCommunicate();

        InitRegistration();

        ROS_INFO("LocalizeMap initialize finish");
    }

    void LocalizeMap::InitParameters() {

        std::string config_file_path = WORK_SPACE_PATH + "/config/matching/matching.yaml";
        config_node = YAML::LoadFile(config_file_path);

        T_cur = Eigen::Matrix4f::Identity();
        T_prev = T_cur;

        //add ------------------------->--------------------------->
        imuPointerFront = 0;
        imuPointerLast = -1;
        imuQueLength = 200;
        timeLaserOdometry = 0;

    }

    void LocalizeMap::InitFilterALL() {
        InitFilter("global_map", global_map_filter_ptr_, config_node);
        InitFilter("local_map", local_map_filter_ptr_, config_node);
        InitFilter("frame", frame_filter_ptr_, config_node);
        box_filter_ptr_ = std::make_shared<BoxFilter>(config_node);
        ROS_INFO("InitFilterALL finish");
    }

    void LocalizeMap::ROSCommunicate() {
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(config_node["lidar_topic"].as<std::string>(), 1,
                                                               &LocalizeMap::cloudHandler, this);
        //add---------> LaserOdometry
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 5,
                                                            &LocalizeMap::laserOdometryHandler, this);
        //add---------> Imu
        //subImu = nh.subscribe<sensor_msgs::Imu> ("/imu/data", 50, &LocalizeMap::imuHandler, this);


        pubRawScan = nh.advertise<sensor_msgs::PointCloud2>(config_node["frame_topic"].as<std::string>(), 1);
        pubNearSubMap = nh.advertise<sensor_msgs::PointCloud2>(config_node["subMap_topic"].as<std::string>(), 1);
        pubGlobalMap = nh.advertise<sensor_msgs::PointCloud2>(config_node["globalMap_topic"].as<std::string>(), 1);
    }

    bool LocalizeMap::InitGlobalMap() {

        string map_path_ = WORK_SPACE_PATH + "/data/finalCloud.pcd";
        CloudData::CLOUD_PTR priorMapCloudRaw(new CloudData::CLOUD());
        if (pcl::io::loadPCDFile<CloudData::POINT>(map_path_, *priorMapCloudRaw) == -1)
            ROS_ERROR ("Couldn't read file %s \n", map_path_.c_str());
        else
            ROS_INFO("Load map with %d points.", priorMapCloudRaw->points.size());

        Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
        trans << 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1;
        pcl::transformPointCloud(*priorMapCloudRaw, *global_map_ptr_, trans);
//        这里是用局部地图滤波器，因为全局地图在这个类里只用于生成局部地图
        local_map_filter_ptr_->Filter(global_map_ptr_, global_map_ptr_);
        has_new_global_map_ = true;

        return true;
    }

    bool LocalizeMap::InitFilter(std::string filter_user, std::shared_ptr<CloudFilterInterface> &filter_ptr,
                                 const YAML::Node &config_node) {
        std::string filter_mothod = config_node[filter_user + "_filter"].as<std::string>();
        LOG(INFO) << "地图匹配" << filter_user << "选择的滤波方法为：" << filter_mothod << std::endl;

        if (filter_mothod == "voxel_filter") {
            filter_ptr = std::make_shared<VoxelFilter>(config_node[filter_mothod][filter_user]);
        } else if (filter_mothod == "no_filter") {
            filter_ptr = std::make_shared<NoFilter>();
        } else {
            LOG(ERROR) << "没有为 " << filter_user << " 找到与 " << filter_mothod << " 相对应的滤波方法!";
            return false;
        }
        return true;
    }

    bool LocalizeMap::InitRegistration() {
        std::string registration_method = config_node["registration_method"].as<std::string>();

        if (registration_method == "NDT") {
            registration_ptr_ = std::make_shared<NDTRegistration>(config_node[registration_method]);
            LOG(INFO) << "地图匹配选择的点云匹配方式为：" << registration_method << std::endl;
        } else {
            LOG(ERROR) << "没找到与 " << registration_method << " 相对应的点云匹配方式!";
            return false;
        }

        ROS_INFO("InitRegistration finish");
        return true;
    }

    void LocalizeMap::cloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg) {

        // 1. Convert ros message to pcl point cloud
        copyPointCloud(laserCloudMsg);

        GuessPose();

        ResetLocalMap();

        update();

        publishCloud();

    }


    //add ----------------------------->laserOdometry------------------------->
    void LocalizeMap::laserOdometryHandler(const nav_msgs::Odometry::ConstPtr& laserOdometry){
        timeLaserOdometry = laserOdometry->header.stamp.toSec();
        double roll, pitch, yaw;
        geometry_msgs::Quaternion geoQuat = laserOdometry->pose.pose.orientation;
        tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w)).getRPY(roll, pitch, yaw);
        transformSum[0] = -pitch;
        transformSum[1] = -yaw;
        transformSum[2] = roll;
        transformSum[3] = laserOdometry->pose.pose.position.x;
        transformSum[4] = laserOdometry->pose.pose.position.y;
        transformSum[5] = laserOdometry->pose.pose.position.z;

        Eigen::Quaterniond q;
        q.x() = geoQuat.z;
        q.y() = -geoQuat.x;
        q.z() = -geoQuat.y;
        q.w() = geoQuat.w;
        Eigen::Matrix3d R = q.normalized().toRotationMatrix();

        //Eigen::Matrix3f R = Eigen::Matrix3f::Identity();//将其赋值为单位矩阵
        //R = Eigen::Matrix3f(geoQuat);    //将四元数转化为旋转矩阵

        Eigen::Matrix4f T_cur_1 = Eigen::Matrix4f::Identity();
        T_cur_1 << R(0,0), R(0,1),R(0,2),transformSum[3],
                   R(1,0), R(1,1),R(1,2),transformSum[4],
                   R(2,0), R(2,1),R(2,2),transformSum[5],
                        0,      0,     0,              1;

        Eigen::Matrix4f velocity_mid1 = T_prev.inverse() * T_cur_1;
    }

    //add -------------------------------->Imu-------------------------------->
    /*void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn){
        double roll, pitch, yaw;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(imuIn->orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        imuPointerLast = (imuPointerLast + 1) % imuQueLength;
        imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
        imuRoll[imuPointerLast] = roll;
        imuPitch[imuPointerLast] = pitch;
    }*/



    void LocalizeMap::copyPointCloud(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg) {

        cloudHeader = laserCloudMsg->header;
        // cloudHeader.stamp = ros::Time::now(); // Ouster lidar users may need to uncomment this line
        pcl::fromROSMsg(*laserCloudMsg, *current_scan_ptr_);
        // Remove Nan points
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*current_scan_ptr_, *current_scan_ptr_, indices);

        frame_filter_ptr_->Filter(current_scan_ptr_, current_scan_ptr_);
        LOG(INFO) << "copyPointCloud finished";
    }

    void LocalizeMap::GuessPose() {

        Eigen::Matrix4f velocity_mid2 = T_prev.inverse() * T_cur;
        Eigen::Matrix4f velocity = (velocity_mid1 + velocity_mid2)/2;

        T_guess = T_cur * velocity;
        T_prev = (velocity_mid1 + velocity_mid2)/2;

        LOG(INFO) << "GuessPose finished";
    }

    bool LocalizeMap::ResetLocalMap() {
        std::vector<float> origin;
        origin.resize(3);
        origin[0] = T_guess(0, 3);
        origin[1] = T_guess(1, 3);
        origin[2] = T_guess(2, 3);

        box_filter_ptr_->SetOrigin(origin);
        box_filter_ptr_->Filter(global_map_ptr_, local_map_ptr_);
        has_new_local_map_ = true;

        LOG(INFO) << "ResetLocalMap finished";
        return true;
    }

    void LocalizeMap::update() {

        registration_ptr_->SetInputTarget(local_map_ptr_);

        registration_ptr_->ScanMatch(current_scan_ptr_, T_guess, current_scan_transformed_ptr_, T_cur);

        LOG(INFO) << "update finished";
    }

    void LocalizeMap::publishCloud() {
        static int frequency = 0;
        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*current_scan_transformed_ptr_, output);   // 转换成ROS下的数据类型 最终通过topic发布
        output.header.stamp = ros::Time::now();
        output.header.frame_id = "slam";
        pubRawScan.publish(output);

        pcl::toROSMsg(*local_map_ptr_, output);   // 转换成ROS下的数据类型 最终通过topic发布
        output.header.stamp = ros::Time::now();
        output.header.frame_id = "slam";
        pubNearSubMap.publish(output);

        if (frequency++ % 10 == 0) {
            pcl::toROSMsg(*global_map_ptr_, output);
            output.header.stamp = ros::Time::now();
            output.header.frame_id = "slam";
            pubGlobalMap.publish(output);
        }
        LOG(INFO) << "publishCloud finished";
    }

    LocalizeMap::~LocalizeMap() {}
};


