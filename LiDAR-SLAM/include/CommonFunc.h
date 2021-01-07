//
// Created on 19-3-27.
//

#ifndef PROJECT_COMMONFUNC_H
#define PROJECT_COMMONFUNC_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <Eigen/Geometry>
#include <tf/transform_broadcaster.h>
#include <vector>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>

#include <opencv2/opencv.hpp>

namespace lidar_localization {

    typedef Eigen::Matrix<double, 6, 1> Vector6d;

    template<typename PointT>
    inline void publishCLoudMsg(ros::Publisher &publisher,
                                const pcl::PointCloud<PointT> &cloud,
                                std::string frameID) {
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(cloud, msg);
        msg.header.frame_id = frameID;

        publisher.publish(msg);
    };

    inline Eigen::Vector3d dcm2rpy(const Eigen::Matrix3d &R) {
        Eigen::Vector3d rpy;
        rpy[1] = atan2(-R(2, 0), sqrt(pow(R(0, 0), 2) + pow(R(1, 0), 2)));
        if (fabs(rpy[1] - M_PI / 2) < 0.00001) {
            rpy[2] = 0;
            rpy[0] = -atan2(R(0, 1), R(1, 1));
        } else {
            if (fabs(rpy[1] + M_PI / 2) < 0.00001) {
                rpy[2] = 0;
                rpy[0] = -atan2(R(0, 1), R(1, 1));
            } else {
                rpy[2] = atan2(R(1, 0) / cos(rpy[1]), R(0, 0) / cos(rpy[1]));
                rpy[0] = atan2(R(2, 1) / cos(rpy[1]), R(2, 2) / cos(rpy[1]));
            }
        }
        return rpy;
    }

    inline Eigen::Matrix3d rpy2dcm(const Eigen::Vector3d &rpy)//(yaw)
    {
        Eigen::Matrix3d R1;
        R1(0, 0) = 1.0;
        R1(0, 1) = 0.0;
        R1(0, 2) = 0.0;
        R1(1, 0) = 0.0;
        R1(1, 1) = cos(rpy[0]);
        R1(1, 2) = -sin(rpy[0]);
        R1(2, 0) = 0.0;
        R1(2, 1) = -R1(1, 2);
        R1(2, 2) = R1(1, 1);

        Eigen::Matrix3d R2;
        R2(0, 0) = cos(rpy[1]);
        R2(0, 1) = 0.0;
        R2(0, 2) = sin(rpy[1]);
        R2(1, 0) = 0.0;
        R2(1, 1) = 1.0;
        R2(1, 2) = 0.0;
        R2(2, 0) = -R2(0, 2);
        R2(2, 1) = 0.0;
        R2(2, 2) = R2(0, 0);

        Eigen::Matrix3d R3;
        R3(0, 0) = cos(rpy[2]);
        R3(0, 1) = -sin(rpy[2]);
        R3(0, 2) = 0.0;
        R3(1, 0) = -R3(0, 1);
        R3(1, 1) = R3(0, 0);
        R3(1, 2) = 0.0;
        R3(2, 0) = 0.0;
        R3(2, 1) = 0.0;
        R3(2, 2) = 1.0;

        return R3 * R2 * R1;
    }


    inline Vector6d toVector6d(Eigen::Matrix4d &matT) {

        Eigen::Matrix3d rot = matT.block(0, 0, 3, 3);
        Eigen::Vector3d angle = dcm2rpy(rot);
        Eigen::Vector3d trans(matT(0, 3), matT(1, 3), matT(2, 3));
        Vector6d pose;
        pose(0) = trans(0);
        pose(1) = trans(1);
        pose(2) = trans(2);
        pose(3) = angle(0) * 180 / M_PI;
        pose(4) = angle(1) * 180 / M_PI;
        pose(5) = angle(2) * 180 / M_PI;
        return pose;
    }

    inline Eigen::Isometry3d rosOdoMsg2Iso3d(nav_msgs::Odometry &odom) {

        Eigen::Quaterniond qua(odom.pose.pose.orientation.w,
                               odom.pose.pose.orientation.x,
                               odom.pose.pose.orientation.y,
                               odom.pose.pose.orientation.z);
        Eigen::Vector3d trans(odom.pose.pose.position.x,
                              odom.pose.pose.position.y,
                              odom.pose.pose.position.z);

        Eigen::Isometry3d pose;
        pose.setIdentity();
        pose.translate(trans);
        pose.rotate(qua);

        return pose;

    }

    inline Eigen::Matrix4f toEigen4fInverse(const Eigen::Matrix4f &Tcw) {
        Eigen::Matrix3f Rcw = Tcw.block(0, 0, 3, 3);
        Eigen::Vector3f tcw = Tcw.block(0, 3, 3, 1);
        Eigen::Matrix3f Rwc = Rcw.transpose();
        Eigen::Vector3f twc = -Rwc * tcw;

        Eigen::Matrix4f Twc = Eigen::Matrix4f::Identity();

        Twc.block(0, 0, 3, 3) = Rwc;
        Twc.block(0, 3, 3, 1) = twc;

        return Twc;
    }


    inline Eigen::Matrix4d rosOdoMsg2Mat4d(const nav_msgs::Odometry &odom) {

        Eigen::Quaterniond qua(odom.pose.pose.orientation.w,
                               odom.pose.pose.orientation.x,
                               odom.pose.pose.orientation.y,
                               odom.pose.pose.orientation.z);
        Eigen::Vector3d trans(odom.pose.pose.position.x,
                              odom.pose.pose.position.y,
                              odom.pose.pose.position.z);

        Eigen::Matrix4d odoPose;
        odoPose.setIdentity();
        odoPose.block(0, 0, 3, 3) = qua.toRotationMatrix();
        odoPose(0, 3) = trans(0);
        odoPose(1, 3) = trans(1);
        odoPose(2, 3) = trans(2);

        return odoPose;

    }

    inline void Mat4d2OdoMsg(Eigen::Matrix4d &pose, nav_msgs::Odometry &poseRos) {

        Eigen::Vector3d transV(pose(0, 3), pose(1, 3), pose(2, 3));
        Eigen::Matrix3d rotMat = pose.block(0, 0, 3, 3);
        Eigen::Quaterniond qua(rotMat);

        poseRos.pose.pose.position.x = transV(0);
        poseRos.pose.pose.position.y = transV(1);
        poseRos.pose.pose.position.z = transV(2);
        poseRos.pose.pose.orientation.x = qua.x();
        poseRos.pose.pose.orientation.y = qua.y();
        poseRos.pose.pose.orientation.z = qua.z();
        poseRos.pose.pose.orientation.w = qua.w();

    }

    inline Eigen::Matrix4d rosGeoMsg2Mat4d(const geometry_msgs::PoseStamped &Pose) {


        Eigen::Quaterniond qua(Pose.pose.orientation.w,
                               Pose.pose.orientation.x,
                               Pose.pose.orientation.y,
                               Pose.pose.orientation.z);
        Eigen::Vector3d trans(Pose.pose.position.x,
                              Pose.pose.position.y,
                              Pose.pose.position.z);

        Eigen::Matrix4d pose;
        pose.setIdentity();
        pose.block(0, 0, 3, 3) = qua.toRotationMatrix();
        pose(0, 3) = trans(0);
        pose(1, 3) = trans(1);
        pose(2, 3) = trans(2);

        return pose;

    };

    inline geometry_msgs::PoseStamped Mat4d2rosGeoMsg(Eigen::Matrix4d &Pose) {


        Eigen::Matrix3d rot = Pose.block(0, 0, 3, 3);

        Eigen::Quaterniond qua(rot);

        geometry_msgs::PoseStamped poseMsg;

        poseMsg.pose.position.x = Pose(0, 3);
        poseMsg.pose.position.y = Pose(1, 3);
        poseMsg.pose.position.z = Pose(2, 3);

        poseMsg.pose.orientation.x = qua.x();
        poseMsg.pose.orientation.y = qua.y();
        poseMsg.pose.orientation.z = qua.z();
        poseMsg.pose.orientation.w = qua.w();

        return poseMsg;

    };

    inline Eigen::Isometry3d Mat4d2Iso3d(Eigen::Matrix4d &mat) {

        Eigen::Isometry3d matIso;
        matIso.setIdentity();

        Eigen::Matrix3d rotMat = mat.block(0, 0, 3, 3);
        Eigen::Vector3d trans(mat(0, 3), mat(1, 3), mat(2, 3));

        matIso.translate(trans);
        matIso.rotate(rotMat);

        return matIso;
    }

    inline Eigen::Matrix4d Iso3d2Mat4d(Eigen::Isometry3d &iso) {

        Eigen::Matrix3d rot = iso.rotation();
        Eigen::Vector3d trans = iso.translation();
        Eigen::Matrix4d mat;
        mat.setIdentity();
        mat.block(0, 0, 3, 3) = rot;
        mat(0, 3) = trans(0);
        mat(1, 3) = trans(1);
        mat(2, 3) = trans(2);

        return mat;

    }

    inline Eigen::Matrix4f Mat4d2Mat4f(Eigen::Matrix4d &mat4d) {

        Eigen::Matrix4f mat4f;

        mat4f << float(mat4d(0, 0)), float(mat4d(0, 1)), float(mat4d(0, 2)), float(mat4d(0, 3)),
                float(mat4d(1, 0)), float(mat4d(1, 1)), float(mat4d(1, 2)), float(mat4d(1, 3)),
                float(mat4d(2, 0)), float(mat4d(2, 1)), float(mat4d(2, 2)), float(mat4d(2, 3)),
                float(mat4d(3, 0)), float(mat4d(3, 1)), float(mat4d(3, 2)), float(mat4d(3, 3));

        return mat4f;
    }

    inline Eigen::Matrix4d Mat4f2Mat4d(Eigen::Matrix4f &mat4f) {

        Eigen::Matrix4d mat4d;

        mat4d << double(mat4f(0, 0)), double(mat4f(0, 1)), double(mat4f(0, 2)), double(mat4f(0, 3)),
                double(mat4f(1, 0)), double(mat4f(1, 1)), double(mat4f(1, 2)), double(mat4f(1, 3)),
                double(mat4f(2, 0)), double(mat4f(2, 1)), double(mat4f(2, 2)), double(mat4f(2, 3)),
                double(mat4f(3, 0)), double(mat4f(3, 1)), double(mat4f(3, 2)), double(mat4f(3, 3));

        return mat4d;
    }

    inline cv::Mat toCVMat(Eigen::Matrix4d &eMat) {

        cv::Mat cvMat(4, 4, CV_32FC1);

        for (int i = 0; i < 4; i++)

            for (int j = 0; j < 4; j++)

                cvMat.at<float>(i, j) = eMat(i, j);

        return cvMat;

    }

    inline Eigen::Matrix4d toEiMat(cv::Mat &cvMat) {

        Eigen::Matrix4d eMat;
        eMat.setIdentity();

        for (int i = 0; i < 4; i++)

            for (int j = 0; j < 4; j++)

                eMat(i, j) = cvMat.at<float>(i, j);

        return eMat;

    }

}

#endif //PROJECT_COMMONFUNC_H
