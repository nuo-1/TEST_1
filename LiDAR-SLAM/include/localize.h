//
// Created by qzj on 2020/12/28.
//

#ifndef SRC_LOCALIZE_H
#define SRC_LOCALIZE_H

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
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include "CommonFunc.h"
#include "cloud_filter/box_filter.hpp"
#include "cloud_filter/voxel_filter.hpp"
#include "cloud_filter/no_filter.hpp"
#include "global_defination/global_defination.h"
#include "glog/logging.h"
#include "registration/ndt_registration.hpp"

// add --------------->------------------->
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>

using namespace std;
using namespace Eigen;
#ifdef CUDA_FOUND
#include <ndt_gpu/NormalDistributionsTransform.h>
#endif

namespace lidar_localization {

    class LocalizeMap {
    private:

        ros::NodeHandle nh;

        ros::Publisher pubRawScan;
        ros::Publisher pubNearSubMap;
        ros::Publisher pubGlobalMap;

        ros::Subscriber subLaserCloud;

        CloudData::CLOUD_PTR local_map_ptr_;
        CloudData::CLOUD_PTR global_map_ptr_;
        CloudData::CLOUD_PTR current_scan_ptr_;
        CloudData::CLOUD_PTR current_scan_transformed_ptr_;

        std::shared_ptr<BoxFilter> box_filter_ptr_;
        std::shared_ptr<CloudFilterInterface> frame_filter_ptr_;
        std::shared_ptr<CloudFilterInterface> local_map_filter_ptr_;
        std::shared_ptr<CloudFilterInterface> global_map_filter_ptr_;

        std::shared_ptr<RegistrationInterface> registration_ptr_;

        YAML::Node config_node;

        Eigen::Matrix4f T_cur;
        Eigen::Matrix4f T_prev;
        Eigen::Matrix4f T_guess;

        std_msgs::Header cloudHeader;

        bool has_new_global_map_;
        bool has_new_local_map_;

        //add ------>------>
        ros::Subscriber subLaserOdometry;
        ros::Subscriber subImu;

        Eigen::Matrix4f velocity_mid1 = Eigen::Matrix4f::Identity();

        float transformSum[6];
        int imuQueLength;
        int imuPointerFront;
        int imuPointerLast;

        double timeLaserOdometry;

    public:


        LocalizeMap();

        ~LocalizeMap();

        void InitParameters();

        bool InitGlobalMap();

        bool ResetLocalMap();

        void GuessPose();

        bool InitFilter(std::string filter_user, std::shared_ptr<CloudFilterInterface> &filter_ptr,
                        const YAML::Node &config_node);

        void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg);

        void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg);

        //add laser odometry
        void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr& laserOdometry);

        //add imu
        //void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn);



        void update();

        void publishCloud();

        void ROSCommunicate();

        bool InitRegistration();

        void InitFilterALL();
    };

}

#endif //SRC_LOCALIZE_H
