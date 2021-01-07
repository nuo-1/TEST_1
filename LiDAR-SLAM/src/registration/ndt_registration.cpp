/*
 * @Description: NDT 匹配模块
 * @Author: Ren Qian
 * @Date: 2020-02-08 21:46:45
 */
#include "registration/ndt_registration.hpp"
#include "glog/logging.h"
#include <ros/ros.h>

namespace lidar_localization {

    NDTRegistration::NDTRegistration(const YAML::Node &node):sumTime(0.f), cnt(0)
    {

#ifdef USE_NDT_OMP
        ndt_ptr_.reset(new pclomp::NormalDistributionsTransform<CloudData::POINT, CloudData::POINT>());
#else
        ndt_ptr_.reset(new pcl::NormalDistributionsTransform<CloudData::POINT, CloudData::POINT>());
#endif

        float res = node["res"].as<float>();
        float step_size = node["step_size"].as<float>();
        float trans_eps = node["trans_eps"].as<float>();
        int max_iter = node["max_iter"].as<int>();

        SetRegistrationParam(res, step_size, trans_eps, max_iter);
    }

    NDTRegistration::NDTRegistration(float res, float step_size, float trans_eps, int max_iter){
#ifdef USE_NDT_OMP
        ndt_ptr_.reset(new pclomp::NormalDistributionsTransform<CloudData::POINT, CloudData::POINT>());
#else
        ndt_ptr_.reset(new pcl::NormalDistributionsTransform<CloudData::POINT, CloudData::POINT>());
#endif
        SetRegistrationParam(res, step_size, trans_eps, max_iter);
    }

    bool NDTRegistration::SetRegistrationParam(float res, float step_size, float trans_eps, int max_iter) {
        ndt_ptr_->setResolution(res);
        ndt_ptr_->setStepSize(step_size);
        ndt_ptr_->setTransformationEpsilon(trans_eps);
        ndt_ptr_->setMaximumIterations(max_iter);

#ifdef USE_NDT_OMP
        ndt_ptr_->setNumThreads(omp_get_max_threads());
//        DIRECT7既快又好，KDtree和PCL一样，DIRECT1最快但转弯的时候不稳定
        ndt_ptr_->setNeighborhoodSearchMethod(pclomp::DIRECT7);
#endif

        LOG(INFO) << "NDT 的匹配参数为：" << std::endl
                  << "res: " << res << ", "
                  << "step_size: " << step_size << ", "
                  << "trans_eps: " << trans_eps << ", "
                  << "max_iter: " << max_iter
                  << std::endl << std::endl;

        return true;
    }

    bool NDTRegistration::SetInputTarget(const CloudData::CLOUD_PTR &input_target) {

        ndt_ptr_->setInputTarget(input_target);
        return true;
    }

    bool NDTRegistration::ScanMatch(const CloudData::CLOUD_PTR &input_source,
                                    const Eigen::Matrix4f &predict_pose,
                                    CloudData::CLOUD_PTR &result_cloud_ptr,
                                    Eigen::Matrix4f &result_pose) {

        ros::Time beginTime = ros::Time::now();

        ndt_ptr_->setInputSource(input_source);
        ndt_ptr_->align(*result_cloud_ptr, predict_pose);
        result_pose = ndt_ptr_->getFinalTransformation();
        bool hasConverged = ndt_ptr_->hasConverged();
        double Score = ndt_ptr_->getFitnessScore(0.5);
        int FinalNumIteration = ndt_ptr_->getFinalNumIteration();

        if (!hasConverged) {
            ROS_ERROR("NDT has not converged!");
        } else {
            ros::Time endTime = ros::Time::now();
            sumTime = sumTime + (endTime - beginTime).toSec();
            cnt = cnt + 1;
            ROS_INFO("Time consumption: aver %f, cur %f seconds. Iteration %d times. Score: %f.",
                     GetAverTime(), (endTime - beginTime).toSec(), FinalNumIteration, Score);
        }

        return true;
    }

    float NDTRegistration::GetAverTime() {
        float re = sumTime / cnt;
        return re;
    }

    float NDTRegistration::GetFitnessScore() {
        return ndt_ptr_->getFitnessScore();
    }

    bool NDTRegistration::GetHasConverged() {
        return ndt_ptr_->hasConverged();
    }

#ifdef CUDA_FOUND
    void LocalizeMap::ndtGPURegistration()
{
    ros::Time beginTime = ros::Time::now();

    gpu::GNormalDistributionsTransform gpu_ndt_solver;
    //cerr<<"1"<<endl;
    gpu_ndt_solver.setTransformationEpsilon(_ndtEpsilon);
    //cerr<<"2"<<endl;
    gpu_ndt_solver.setStepSize(_ndtStepSize);
    //cerr<<"3"<<endl;
    gpu_ndt_solver.setResolution(_ndtGridResolution);
    //cerr<<"4"<<endl;
    gpu_ndt_solver.setMaximumIterations(_ndtMaxIterations);
    //cerr<<"5"<<endl;

    //Load clouds
    gpu_ndt_solver.setInputSource(current_scan_ptr_);
    //cerr<<"6"<<endl;
    gpu_ndt_solver.setInputTarget(nearSubMap);
    //cerr<<"7"<<endl;

    gpu_ndt_solver.align(T_cur);
    //cerr<<"8"<<endl;
    bool hasConverged = gpu_ndt_solver.hasConverged();
    //cerr<<"9"<<endl;
    int FinalNumIteration = gpu_ndt_solver.getFinalNumIteration();
    //cerr<<"10"<<endl;
    double Score = gpu_ndt_solver.getFitnessScore(0.1);

    if (!hasConverged){
        ROS_ERROR("NDT has not converged!");
    }
    else{

        T_map_scan = gpu_ndt_solver.getFinalTransformation();
        T_cur = T_map_scan;

        ros::Time endTime = ros::Time::now();
        ROS_INFO("Time consumption of NDT matching: %f seconds. Iteration %d times. Score: %f."
        ,(endTime - beginTime).toSec(), FinalNumIteration,Score);
    }
}
#endif

}