cmake_minimum_required(VERSION 2.8)
project(lidar_slam)

# 一定要加-march=native， 不然使用pcl的一些函数会报错
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -march=native")
set ( CMAKE_WARN_DEPRECATED 0 )

find_package(catkin REQUIRED COMPONENTS
        tf
        roscpp
        rospy

        pcl_ros
        pcl_conversions

        std_msgs
        sensor_msgs
        geometry_msgs
        )
catkin_package(
)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS io registration common filter kdtree)
#
find_package(Boost REQUIRED COMPONENTS regex thread system filesystem)
add_definitions(-DBOOST_NO_CXX11_SCOPED_ENUMS)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include

        ${PCL_INCLUDE_DIRS}

        ${Boost_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${catkin_INCLUDE_DIRS}
)

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_DIR)

add_library(${PROJECT_NAME} SHARED
        ${SRC_DIR}
        )


target_link_libraries(${PROJECT_NAME}
        ${Boost_LIBRARIES}
        ${EIGEN3_LIBS}

        ${PCL_LIBRARIES}
        ${catkin_LIBRARIES}
        )


#add_executable(dataLoader nodecpp/dataLoader.cpp)
#target_link_libraries(dataLoader ${PROJECT_NAME})
#
#add_executable(mapping nodecpp/mapping.cpp)
#target_link_libraries(mapping ${PROJECT_NAME})

add_executable(localize_map nodecpp/localize_map.cpp)
target_link_libraries(localize_map ${PROJECT_NAME})


