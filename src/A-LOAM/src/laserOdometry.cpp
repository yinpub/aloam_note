// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cmath>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <queue>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "lidarFactor.hpp"

#define DISTORTION 0


int corner_correspondence = 0, plane_correspondence = 0;

constexpr double SCAN_PERIOD = 0.1; //扫描的周期  一次0.1s
constexpr double DISTANCE_SQ_THRESHOLD = 25;//阈值  当进行数据关联的时候，Loam使用了这个阈值。这个阈值用以判别这组特征点（当前点以及被关联的点）是否可用于构建残存 这里设置为25，其对应实际距离为5。
constexpr double NEARBY_SCAN = 2.5;//激光束临近判别范围

int skipFrameNum = 5;
bool systemInited = false;

double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;

pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());//点特征 曲率特别大
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());//点特征 曲率大   放在全局区
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());//面特征 曲率特别小
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());//面特征 曲率小

pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

int laserCloudCornerLastNum = 0;
int laserCloudSurfLastNum = 0;

// Odometry里的位姿更新策略为：每次通过两帧计算一个相对变换，用上一时刻的世界坐标乘当前时刻的变换得到这一时刻的世界坐标    Transformation from current frame to world frame
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);//对于里程计而言，相对于世界坐标系的旋转 
Eigen::Vector3d t_w_curr(0, 0, 0);//对于里程计而言，相对于世界坐标系的位移

// q_curr_last(x, y, z, w), t_curr_last
double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};

Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);//上一时刻两帧间相对姿态变换
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t); //上一时刻两帧间相对位移变换

std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;
std::mutex mBuf;

//运动去畸变  调整点云的位置 undistort lidar point
//输入:
//  pi:输入的激光特征点，坐标系为雷达坐标系
//  po:存放输出的激光点
void TransformToStart(PointType const *const pi, PointType *const po)
{
    //1.确定插值比例  s也就是这一个点云采集到的时间，处于整个扫描周期的位置。    interpolation ratio
    double s;
        //1.1把点云补偿到起始位置
    if (DISTORTION)
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD; //在scanRegistration文件可知，我们将intensity的小数部分设为了该点云相对于整个扫描的时间
        //1.2把点云补偿到终止位置
    else
        s = 1.0;
    //2.点云补偿
        //2.1插值求位姿
            //2.1.1使用球面插值求出旋转四元数（在这个雷达周期内，从采集到当前点时刻，旋转到采集结束的时刻）
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr); //（插值占比，上一时刻的姿态变化）         Identity生成的四元数是[0,0,0,1] 虚数部分是1  经过slerp后示例:【0.000511318，-0.000310684，-0.000544876，1】
            //2.1.2使用线性插值求出平移向量
    Eigen::Vector3d t_point_last = s * t_last_curr;
        //2.2求出点新的位置（投影到最后一刻）
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;
        //2.3将去除畸变后的位置重新赋值
    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}

//运动去畸变 transform all lidar points to the start of the next frame
void TransformToEnd(PointType const *const pi, PointType *const po)
{
    // undistort point first
    pcl::PointXYZI un_point_tmp;
    TransformToStart(pi, &un_point_tmp);

    Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
    Eigen::Vector3d point_end = q_last_curr.inverse() * (un_point - t_last_curr);

    po->x = point_end.x();
    po->y = point_end.y();
    po->z = point_end.z();

    //Remove distortion time info
    po->intensity = int(pi->intensity);
}

void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2)
{
    mBuf.lock();
    cornerSharpBuf.push(cornerPointsSharp2);
    mBuf.unlock();
}

void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2)
{
    mBuf.lock();
    cornerLessSharpBuf.push(cornerPointsLessSharp2);
    mBuf.unlock();
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2)
{
    mBuf.lock();
    surfFlatBuf.push(surfPointsFlat2);
    mBuf.unlock();
}

void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2)
{
    mBuf.lock();
    surfLessFlatBuf.push(surfPointsLessFlat2);
    mBuf.unlock();
}

//receive all point cloud
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullPointsBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}



int main(int argc, char **argv)
{
//1)ROS初始化
    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle nh;
//2)加载配置文件  
    //2.1记载    //TODO 不知道这个参数的含义
    nh.param<int>("mapping_skip_frame", skipFrameNum, 2);
    //2.2打印
    printf("Mapping %d Hz \n", 10 / skipFrameNum);
//3)实例化订阅与发布
    //3.1从scanRegistration节点订阅
    ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100, laserCloudSharpHandler);//点特征（曲率特别大）

    ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, laserCloudLessSharpHandler);//点特征（曲率大）

    ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, laserCloudFlatHandler);//面特征（曲率特别小）

    ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, laserCloudLessFlatHandler);//面特征（曲率小）

    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100, laserCloudFullResHandler);//全部点云

    //3.2发布
    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);

    ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);

    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);

    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);

    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);


//3)主循环
    //3.1预设变量
    nav_msgs::Path laserPath;
    int frameCount = 0;
    //3.2设置循环频率
    ros::Rate rate(100);
    //3.3进入循环
    while (ros::ok())
    {
        //3.3.1触发回调
        ros::spinOnce();
        //3.3.2当5个缓存都拿到订阅的数据，进入处理。
        if (!cornerSharpBuf.empty() && !cornerLessSharpBuf.empty() &&
            !surfFlatBuf.empty() && !surfLessFlatBuf.empty() &&
            !fullPointsBuf.empty())
        {
            //2.1保证消息时间同步
                //2.1.1取出5个数据的时间戳
            timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
            timeCornerPointsLessSharp = cornerLessSharpBuf.front()->header.stamp.toSec();
            timeSurfPointsFlat = surfFlatBuf.front()->header.stamp.toSec();
            timeSurfPointsLessFlat = surfLessFlatBuf.front()->header.stamp.toSec();
            timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();
                //2.1.2当时间戳不一致，就报error
            if (timeCornerPointsSharp != timeLaserCloudFullRes ||
                timeCornerPointsLessSharp != timeLaserCloudFullRes ||
                timeSurfPointsFlat != timeLaserCloudFullRes ||
                timeSurfPointsLessFlat != timeLaserCloudFullRes)
            {
                printf("unsync messeage!");
                ROS_BREAK();
            }
            //2.2ROS转PCL
                //2.2.1上锁
            mBuf.lock();
                //2.2.2数据提取与转换
                    //2.1点特征（曲率特别大）
                        //2.1.1清除上一次的cornerPointsSharp
            cornerPointsSharp->clear();
                        //2.1.2取出数据，并转换格式或者说给cornerPointsSharp赋值。  之前初始化cornerPointsSharp是在全局区域。
            pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);
                        //2.1.3在缓存中踢出已取出的数据
            cornerSharpBuf.pop();
                    //2.2点特征（曲率大）
            cornerPointsLessSharp->clear();
            pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);
            cornerLessSharpBuf.pop();
                    //2.3面特征（曲率特别小）
            surfPointsFlat->clear();
            pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
            surfFlatBuf.pop();
                    //2.4面特征（曲率小）
            surfPointsLessFlat->clear();
            pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
            surfLessFlatBuf.pop();
                    //2.5全部点云
            laserCloudFullRes->clear();
            pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
            fullPointsBuf.pop();
                //2.2.3解锁
            mBuf.unlock();
            
            //2.3特征点匹配位姿估计  
            TicToc t_whole;
                //2.3.1跳过第一帧，并置位系统初始化状态 initializing
            if (!systemInited)
            {
                systemInited = true;
                std::cout << "Initialization finished \n";
            }
                //2.3.2从第二帧开始处理
            else
            {
                    //2.1统计特征点数量
                int cornerPointsSharpNum = cornerPointsSharp->points.size(); //点特征数量（曲率特别大）
                int surfPointsFlatNum = surfPointsFlat->points.size();       //面特征数量（曲率特别小）
                    
                TicToc t_opt;
                    //2.2进行两次优化 
                for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter)
                {
                        //2.2.1构造优化问题
                            //1.1变量置0
                    corner_correspondence = 0;//参与优化的点特征数，也就是点特征残差项数量
                    plane_correspondence = 0;//参与优化的面特征数，也就是面特征残差项数量
                            //1.2创建核函数
                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                            //1.3创建待优化变量 要优化的变量是位姿，位姿的旋转部分不满足加法，所以无法直接设置一个Eigen或opencv的Mat位姿矩阵为优化变量。针对这个问题，ceres推出了专用变量LocalParameterization。
                    ceres::LocalParameterization *q_parameterization =
                        new ceres::EigenQuaternionParameterization();
                            //1.4实例化优化问题
                    ceres::Problem::Options problem_options;
                    ceres::Problem problem(problem_options);
                            //1.5加入变量
                    problem.AddParameterBlock(para_q, 4, q_parameterization);
                    problem.AddParameterBlock(para_t, 3);
                            //1.6定义变量
                    pcl::PointXYZI pointSel;//用以存放去掉畸变后的点云
                    std::vector<int> pointSearchInd;//用以存放索引； 当进行特征点搜索时，我们需要用这个向量记录，上一帧点云中哪些特征点是被检索出来的。
                    std::vector<float> pointSearchSqDis;//用以存放搜索出的近邻点与参考特征点的欧式距离；

                    TicToc t_data;
                        //2.2.2遍历点特征（曲率特别大）  进行特征点数据关联 find correspondence for corner features
                    for (int i = 0; i < cornerPointsSharpNum; ++i)
                    {
                            //2.1点云去畸变
                        TransformToStart(&(cornerPointsSharp->points[i]), &pointSel);//(要去畸变的点云地址，去掉畸变后点云存放的地址)
                            //2.2最临近搜索最近邻点
                         kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);//(参考特征点，搜索特征点数量，向量存放在上一帧对应点的索引,向量存放这两个点的欧式距离) 如果搜索特征点数量为4，那就会搜索出4个特征点
                        int closestPointInd = -1, minPointInd2 = -1;
                            //2.3当特征点的欧式距离小于阈值，认为特征点有效，并在近临特征点附近搜索次近邻点
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                        {
                                //2.3.1获取近邻点在整个上一帧点云的索引
                            closestPointInd = pointSearchInd[0];
                                //2.3.2获取近邻点属于哪个激光线
                            int closestPointScanID = int(laserCloudCornerLast->points[closestPointInd].intensity);

                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;//次临近点的筛选阈值。这个阈值会不断的更新   这里第一次，先给minPointSqDis2赋值为初始阈值
                                //2.3.3在两个方向，寻找1个次近邻点          search in the direction of increasing scan line
                                    //3.1在激光线束增大的方向寻找次近邻点
                            for (int j = closestPointInd + 1; j < (int)laserCloudCornerLast->points.size(); ++j)
                            {
                                        //3.1.1如果在同一个激光线束上或减小的方向，跳出本次循环 if in the same scan line, continue
                                if (int(laserCloudCornerLast->points[j].intensity) <= closestPointScanID)
                                    continue;

                                        //3.1.2如果超越了临近的范围，跳出整个循环 if not in nearby scans, end the loop
                                if (int(laserCloudCornerLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;
                                        //3.1.3计算次近邻点和目标点的欧式距离
                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                        (laserCloudCornerLast->points[j].z - pointSel.z);
                                        //3.1.4当距离小于阈值，认为次近邻点有效
                                if (pointSqDis < minPointSqDis2)
                                {
                                            //4.1更新阈值为当前两点的距离
                                    minPointSqDis2 = pointSqDis;
                                            //4.2更新近邻点索引
                                    minPointInd2 = j;
                                }
                            }

                                    //3.2在激光线束减小的方向寻找次近邻点 search in the direction of decreasing scan line
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                        //3.2.1如果在同一个激光线束上或增大的方向，跳出本次循环 if in the same scan line, continue
                                if (int(laserCloudCornerLast->points[j].intensity) >= closestPointScanID)
                                    continue;

                                        //3.2.2如果超越了临近的范围，跳出整个循环 if not in nearby scans, end the loop
                                if (int(laserCloudCornerLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;
                                        //3.2.3计算次近邻点和目标点的欧式距离
                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                        (laserCloudCornerLast->points[j].z - pointSel.z);
                                        //3.2.4当距离小于阈值，认为次近邻点有效
                                if (pointSqDis < minPointSqDis2)
                                {
                                            //4.1更新阈值为当前距离 find nearer point
                                    minPointSqDis2 = pointSqDis;\
                                            //4.2更新近邻点索引
                                    minPointInd2 = j;
                                }
                            }
                        }
                            //2.4当两个近临特征点（分别是通过KDtree求出的近邻点和对两边遍历求得的此邻近点）有效，构造优化问题
                        if (minPointInd2 >= 0) // both closestPointInd and minPointInd2 is valid
                        {
                                //2.4.1取出特征点
                                    //1.1取出当前帧的特征点
                            Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                                                       cornerPointsSharp->points[i].y,
                                                       cornerPointsSharp->points[i].z);
                                    //1.2取出上一帧的近邻点
                            Eigen::Vector3d last_point_a(laserCloudCornerLast->points[closestPointInd].x,
                                                         laserCloudCornerLast->points[closestPointInd].y,
                                                         laserCloudCornerLast->points[closestPointInd].z);
                                    //1.3取出上一帧的次近邻点
                            Eigen::Vector3d last_point_b(laserCloudCornerLast->points[minPointInd2].x,
                                                         laserCloudCornerLast->points[minPointInd2].y,
                                                         laserCloudCornerLast->points[minPointInd2].z);
                                //2.4.2运动补偿系数
                            double s;
                            if (DISTORTION)
                                s = (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) / SCAN_PERIOD;
                            else
                                s = 1.0;
                                //2.4.3添加残差
                            ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                //2.4.4残差项+1
                            corner_correspondence++;
                        }
                    }




                        //2.2.3遍历面特征    find correspondence for plane features
                    for (int i = 0; i < surfPointsFlatNum; ++i)
                    {
                            //3.1点云去畸变
                        TransformToStart(&(surfPointsFlat->points[i]), &pointSel);
                            //3.2最临近搜索最近邻点
                        kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
                        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                            //3.3当特征点的欧式距离小于阈值，认为特征点有效，并在近临特征点附近搜索次近邻点
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                        {
                                //3.3.1获取近邻点在整个上一帧点云的索引
                            closestPointInd = pointSearchInd[0];
                                //3.3.2获取近邻点属于哪个激光线 get closest point's scan ID
                            int closestPointScanID = int(laserCloudSurfLast->points[closestPointInd].intensity);
                            
                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;
                                //3.3.3在两个方向，寻找1个次近邻点   search in the direction of increasing scan line
                                    //3.1在激光线束增大的方向寻找次近邻点
                            for (int j = closestPointInd + 1; j < (int)laserCloudSurfLast->points.size(); ++j)
                            {
                                        //3.1.1如果超越了近邻的范围，跳出整个循环if not in nearby scans, end the loop
                                if (int(laserCloudSurfLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;
                                        //3.1.2计算次近邻点和目标点的欧式距离
                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                        //3.1.3寻找两个次近邻点
                                            //3.1如果在同一个激光线束上或减小的方向 if in the same or lower scan line
                                if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                            //3.2如果在增大的方向 if in the higher scan line
                                else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                                    //3.2在激光线束减小的方向寻找次近邻点 search in the direction of decreasing scan line
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if not in nearby scans, end the loop
                                if (int(laserCloudSurfLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                // if in the same or higher scan line
                                if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                else if (int(laserCloudSurfLast->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    // find nearer point
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }
                                    //3.3当3个近邻点有效时，构造优化问题
                            if (minPointInd2 >= 0 && minPointInd3 >= 0)
                            {

                                Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
                                                            surfPointsFlat->points[i].y,
                                                            surfPointsFlat->points[i].z);
                                Eigen::Vector3d last_point_a(laserCloudSurfLast->points[closestPointInd].x,
                                                                laserCloudSurfLast->points[closestPointInd].y,
                                                                laserCloudSurfLast->points[closestPointInd].z);
                                Eigen::Vector3d last_point_b(laserCloudSurfLast->points[minPointInd2].x,
                                                                laserCloudSurfLast->points[minPointInd2].y,
                                                                laserCloudSurfLast->points[minPointInd2].z);
                                Eigen::Vector3d last_point_c(laserCloudSurfLast->points[minPointInd3].x,
                                                                laserCloudSurfLast->points[minPointInd3].y,
                                                                laserCloudSurfLast->points[minPointInd3].z);

                                double s;
                                if (DISTORTION)
                                    s = (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity)) / SCAN_PERIOD;
                                else
                                    s = 1.0;
                                ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                plane_correspondence++;
                            }
                        }
                    }
                        //2.3.4打印调试信息
                    //printf("coner_correspondance %d, plane_correspondence %d \n", corner_correspondence, plane_correspondence);
                    printf("data association time %f ms \n", t_data.toc());

                    if ((corner_correspondence + plane_correspondence) < 10)
                    {
                        printf("less correspondence! *************************************************\n");
                    }
                        //2.3.5优化
                            //5.1配置优化器
                    TicToc t_solver;
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_QR;
                    options.max_num_iterations = 4;
                    options.minimizer_progress_to_stdout = false;
                    ceres::Solver::Summary summary;
                            //5.2求解最小二乘法
                    ceres::Solve(options, &problem, &summary);
                    printf("solver time %f ms \n", t_solver.toc());
                }
                    
                    //2.3打印优化两次的时间
                printf("optimization twice time %f \n", t_opt.toc());
                    //2.4求解当前帧的位姿
                t_w_curr = t_w_curr + q_w_curr * t_last_curr;//相对于世界坐标系的位移向量
                q_w_curr = q_w_curr * q_last_curr;//相对于世界坐标系的旋转
            }
            
            TicToc t_pub;

            //2.4发布数据publish odometry
                //2.4.1封装odometry（laser_odom相对于camera_init的变换）
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "/camera_init";
            laserOdometry.child_frame_id = "/laser_odom";
            laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
            laserOdometry.pose.pose.orientation.x = q_w_curr.x();
            laserOdometry.pose.pose.orientation.y = q_w_curr.y();
            laserOdometry.pose.pose.orientation.z = q_w_curr.z();
            laserOdometry.pose.pose.orientation.w = q_w_curr.w();
            laserOdometry.pose.pose.position.x = t_w_curr.x();
            laserOdometry.pose.pose.position.y = t_w_curr.y();
            laserOdometry.pose.pose.position.z = t_w_curr.z();
            pubLaserOdometry.publish(laserOdometry);
                //2.4.2封装pose
            geometry_msgs::PoseStamped laserPose;
            laserPose.header = laserOdometry.header;
            laserPose.pose = laserOdometry.pose.pose;
            laserPath.header.stamp = laserOdometry.header.stamp;
            laserPath.poses.push_back(laserPose);
            laserPath.header.frame_id = "/camera_init";
            pubLaserPath.publish(laserPath);

            //2.5另一种去畸变方式，为0时不使用 transform corner features and plane features to the scan end point
            if (0)
            {
                int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
                for (int i = 0; i < cornerPointsLessSharpNum; i++)
                {
                    TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
                }

                int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
                for (int i = 0; i < surfPointsLessFlatNum; i++)
                {
                    TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
                }

                int laserCloudFullResNum = laserCloudFullRes->points.size();
                for (int i = 0; i < laserCloudFullResNum; i++)
                {
                    TransformToEnd(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
                }
            }
            //2.6将当前帧的特征点置为上一帧的特征点 （为下一帧做准备）
            pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
            cornerPointsLessSharp = laserCloudCornerLast;
            laserCloudCornerLast = laserCloudTemp;

            laserCloudTemp = surfPointsLessFlat;
            surfPointsLessFlat = laserCloudSurfLast;
            laserCloudSurfLast = laserCloudTemp;

            laserCloudCornerLastNum = laserCloudCornerLast->points.size();
            laserCloudSurfLastNum = laserCloudSurfLast->points.size();

            // std::cout << "the size of corner last is " << laserCloudCornerLastNum << ", and the size of surf last is " << laserCloudSurfLastNum << '\n';

            kdtreeCornerLast->setInputCloud(laserCloudCornerLast);//将当前帧（点特征 曲率大）设为上一帧，用于下一帧匹配（数据关联）  点特征 曲率大
            kdtreeSurfLast->setInputCloud(laserCloudSurfLast);//将当前帧（面特征 曲率小）设为上一帧
            
            //2.7当经过固定帧数后，发布点云给后端数据
            if (frameCount % skipFrameNum == 0)
            {
                //2.7.1变量置0
                frameCount = 0;
                //2.7.2发布点云
                    //2.1发布点特征（曲率大）
                sensor_msgs::PointCloud2 laserCloudCornerLast2;
                pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
                laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudCornerLast2.header.frame_id = "/camera";
                pubLaserCloudCornerLast.publish(laserCloudCornerLast2);
                    //2.2发布面特征（曲率小）
                sensor_msgs::PointCloud2 laserCloudSurfLast2;
                pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
                laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudSurfLast2.header.frame_id = "/camera";
                pubLaserCloudSurfLast.publish(laserCloudSurfLast2);
                    //2.3发布全部点云
                sensor_msgs::PointCloud2 laserCloudFullRes3;
                pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
                laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudFullRes3.header.frame_id = "/camera";
                pubLaserCloudFullRes.publish(laserCloudFullRes3);
            }
            //2.8打印信息
            printf("publication time %f ms \n", t_pub.toc());
            printf("whole laserOdometry time %f ms \n \n", t_whole.toc());
            if(t_whole.toc() > 100)
                ROS_WARN("odometry process over 100ms");
            //2.9处理帧数+1
            frameCount++;
        }
        //3.3.3调节触发频率
        rate.sleep();
    }
    return 0;
}