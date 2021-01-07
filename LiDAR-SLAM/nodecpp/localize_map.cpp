//
// Created by qzj on 2020/12/27.
//

#include "ros/ros.h"
#include <vector>
#include "localize.h"
#include "global_defination/global_defination.h"
#include "glog/logging.h"

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {

    google::InitGoogleLogging(argv[0]);
    ros::init(argc, argv, "localize_map");

    FLAGS_log_dir = lidar_localization::WORK_SPACE_PATH + "/Log";
    FLAGS_alsologtostderr = false;
    FLAGS_colorlogtostderr = true;
    FLAGS_log_prefix = true;
    FLAGS_logbufsecs = 0;
    ROS_INFO("\033[1;32m---->\033[0m localize_map Started.");

    lidar_localization::LocalizeMap localizeMap;

    ros::spin();

    return 0;
}




