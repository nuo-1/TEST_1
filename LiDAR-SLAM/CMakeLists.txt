cmake_minimum_required(VERSION 2.8)
project(lidar_localization)

# 一定要加-march=native， 不然使用pcl的一些函数会报错
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 ")
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()
set(CMAKE_WARN_DEPRECATED 0)

find_package(Eigen3 REQUIRED)
#find_package(OpenCV REQUIRED)
find_package(GTSAM REQUIRED QUIET)
find_package(PCL REQUIRED QUIET)
find_package(OpenCV REQUIRED QUIET)
#find_package(PCL REQUIRED COMPONENTS io registration common filter kdtree)
#
find_package(Boost REQUIRED COMPONENTS regex thread system filesystem)
add_definitions(-DBOOST_NO_CXX11_SCOPED_ENUMS)
#find_package(CUDA)
#set(CUDA_FOUND 0)
#if (CUDA_FOUND)
#    message("CUDA_FOUND")
#    add_subdirectory(Thirdparty/ndt_gpu)
#    add_definitions(-DCUDA_FOUND)
#    set(ndtcompon ndt_gpu)
#else ()
set(ndtcompon)
#endif ()

find_package(catkin REQUIRED COMPONENTS
        tf
        roscpp
        rospy
        cv_bridge
        image_transport

        pcl_ros
        pcl_conversions

        nav_msgs
        std_msgs
        sensor_msgs
        geometry_msgs

        cloud_msgs

        ${ndtcompon}
        )
#add_subdirectory(Thirdparty/ndt_omp)
#add_definitions(-DUSE_NDT_OMP)
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS cloud_msgs
  DEPENDS PCL
)



file(GLOB_RECURSE ALL_SRCS "*.cpp")
file(GLOB_RECURSE NODE_SRCS "nodecpp/*.cpp")
file(GLOB_RECURSE THIRD_PARTY_SRCS "Thirdparty/*.cpp")
list(REMOVE_ITEM ALL_SRCS ${NODE_SRCS})
list(REMOVE_ITEM ALL_SRCS ${THIRD_PARTY_SRCS})

add_library(${PROJECT_NAME} SHARED
        ${ALL_SRCS}
        )

include(cmake/glog.cmake)
include(cmake/YAML.cmake)
include(cmake/global_defination.cmake)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${catkin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIRS}
#        ${PROJECT_SOURCE_DIR}/Thirdparty/ndt_omp/include
)

target_link_libraries(${PROJECT_NAME}
        ${Boost_LIBRARIES}
        ${EIGEN3_LIBS}
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
#        ${PROJECT_SOURCE_DIR}/Thirdparty/ndt_omp/lib/libndt_omp.so
        )

#add_executable(dataLoader nodecpp/dataLoader.cpp)
#target_link_libraries(dataLoader ${PROJECT_NAME})
#
#add_executable(mapping nodecpp/mapping.cpp)
#target_link_libraries(mapping ${PROJECT_NAME})

add_executable(imageProjection nodecpp/imageProjection.cpp)
add_dependencies(imageProjection ${catkin_EXPORTED_TARGETS} cloud_msgs_gencpp)
target_link_libraries(imageProjection ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(featureAssociation nodecpp/featureAssociation.cpp)
add_dependencies(featureAssociation ${catkin_EXPORTED_TARGETS} cloud_msgs_gencpp)
target_link_libraries(featureAssociation ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(localize_map nodecpp/localize_map.cpp)
target_link_libraries(localize_map ${PROJECT_NAME})


#if (CUDA_FOUND)
#    target_include_directories(localize_map PRIVATE ${CUDA_INCLUDE_DIRS})
#endif ()


