##qsy's note
#### 初始化catkin工作空间
    $ mkdir -p ~/catkin_ws/src
    $ cd ~/catkin_ws/
    $ catkin_make #初始化工作空间

#### package创建
    /catkin_ws/src$ catkin_create_pkg test_pkg roscpp rospy std_msgs
填充　CMakeLists.txt　和　package.xml,　并且将依赖项填进了这两个文件中


------
#### CMakeLists.txt / Boost
    aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_DIR)
    add_library(${PROJECT_NAME} SHARED
        ${SRC_DIR}
        )
    
    target_link_libraries(${PROJECT_NAME}
            ${Boost_LIBRARIES}
            )
        
    add_executable(kitti2pcd src/kitti2pcd.cpp)
    target_link_libraries(kitti2pcd ${PROJECT_NAME})
    
#### package.xml
    <build_depend>
    <exec_depend>std_msgs</exec_depend>
- 第二版：run_depend --> exec_depend


#### run.launch
    <node pkg="lidar_slam" type="kitti2pcd"    name="kitti2pcd"    output="screen"/>
- type = 节点名字 = 可执行文件名字
- name = 命名空间
    
#### my.rviz 
- 第一次运行rviz 配置后可自动生成 .rviz文件


-------
#### 启动node
    src$ roscore
    src$ rosrun pkg_name node_name
    或
    src$ roslaunch lidar_slam run.launch

#### 终端
    $ gedit ~/.bashrc
    
    source /home/qsy-5208/Documents/LeGO/LeGO-LOAM-kitti/catkin_ws/devel/setup.bash

#### Topic确认 / Debug
    $ rostopic echo 


-----
#### 时间同步
错误信息（from rostopic）：
    
    WARNING: no messages received and simulated time is active.
    Is /clock being published?
    
解决：

    run.launch ———— <param name="/use_sim_time" value="false" />   
    .cpp ———— output.header.stamp = ros::Time::now();

------
#### Github
~/Documents/LeGO/LeGO-LOAM-kitti/catkin_ws/src/lidar_slam$

    $ git reset
    $ git add -A
    $ git status
    $ git commit -m ""init
    
    $ git remote add origin https://github.com/QinShuying/LeGO-LOAM-kitti.git
    $ git push origin master

------
#### Edit Configration
若在CMake中 add_executable( .cpp)
但Edit Configration找不到，则手动添加

    Edit Configration --> '+' --> CMake Applilcation --> Target 