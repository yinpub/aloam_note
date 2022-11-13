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
#include <vector>
#include <string>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using std::atan2;
using std::cos;
using std::sin;

const double scanPeriod = 0.1;

const int systemDelay = 0; 
int systemInitCount = 0;
bool systemInited = false; //系统初始化状态
int N_SCANS = 0;//激光雷达线数
float cloudCurvature[400000];//存储各个点的曲率
int cloudSortInd[400000];
int cloudNeighborPicked[400000];//存储各个点是否为特征点
int cloudLabel[400000];

bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]); }

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubRemovePoints;
std::vector<ros::Publisher> pubEachScan;

bool PUB_EACH_LINE = false; // 是否发布每条线扫

double MINIMUM_RANGE = 0.1;  //不接收激光雷达为中心，半径为0.1m的球内的点云

//去除点云中有效范围外的点（近点）     
//输入:
//  cloud_in：
//  cloud_out：
//  thres：无效的半径范围
//这里是函数模板
template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, float thres)
{
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;

    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size())
    {
        cloud_out.points.resize(j);
    }

    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}
//订阅点云的回调函数
//输入:
//  laserCloudMsg：雷达发布的原始点云
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
//1)判断系统是否进行了初始化。此处判断语句是为了延迟，确保IMU数据可以跟上（A loam版本不再使用IMU）
    if (!systemInited)
    { 
    //1.1初始化变量systemInitCount+1
        systemInitCount++;
    //1.2当systemInitCount>初始化阀值，则进行初始化
        if (systemInitCount >= systemDelay) //此时systemDelay为0
        {
            systemInited = true;
        }
    //1.3没有达到阈值时跳出
        else
            return;
    }
//2)新建变量
    TicToc t_whole;
    TicToc t_prepare;//用以统计预处理时间
    std::vector<int> scanStartInd(N_SCANS, 0);//
    std::vector<int> scanEndInd(N_SCANS, 0);//创建N_SCANS长的向量并赋值为0
//3)移出无效点云
    //3.1将ros点云转为pcl点云格式
    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
    std::vector<int> indices;
    //3.2移出NAN点
    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
    //3.3去除点云中距离小于有效范围的点
    removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);

//4)计算起始点和结束点与x轴的夹角   (atan2获取得是点与X轴的夹角，此处我们想获取的是，雷达的这个点与X轴的角度。但雷达是顺时旋转的。比如：当雷达转过45度采集到一个点，用atan2获取得是-45度。当取-atan2就可以得到45度，另注意：atan2是弧度制)   对于这里的一些思考：在雷达坐标系下，第一个点云数据的y是不是0？  答：雷达可能已经转了一段时间，才开始记录数据。
    //4.1获取点云数量
    int cloudSize = laserCloudIn.points.size();
    //4.2第一个点云的位置
    float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
    //4.3最后一个点云的位置
    float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                          laserCloudIn.points[cloudSize - 1].x) +
                   2 * M_PI; 
    //4.4出现异常则进行修正
    if (endOri - startOri > 3 * M_PI)
    {
        endOri -= 2 * M_PI;
    }
    else if (endOri - startOri < M_PI)
    {
        endOri += 2 * M_PI;
    }
//5)计算点的scanID以及时间
    //5.1建立变量
    bool halfPassed = false;//扫描是否过半
    int count = cloudSize;//总点数
    PointType point;//用以从雷达点云中获取单个点
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);//用以存储经过我们下面算法处理过的点云
    //5.2遍历点云
    for (int i = 0; i < cloudSize; i++)
    {   
    //5.3获取点云位置
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;
    //5.4计算垂直俯仰角
        float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
    //5.5找到激光点对应的scanID 对不同的雷达型号做不同的处理
        int scanID = 0;
        //5.5.1当雷达为16线时
        if (N_SCANS == 16)
        {
            //1.1求出该点所在的scanID。    
            scanID = int((angle + 15) / 2 + 0.5);//在竖直方向上，每一次发射出的激光，采集到各点云之间的角度是固定的。每个激光的夹脚是2度，总共16线，那么0-15每一个序号对应一根激光。
            //1.2当得出的scanID不在正常范围内，认为这个点无效，总点数-1
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        //5.5.2当雷达为32线时
        else if (N_SCANS == 32)
        {
            scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        //5.5.3当雷达为64线时
        else if (N_SCANS == 64)
        {   
            if (angle >= -8.83)
                scanID = int((2 - angle) * 3.0 + 0.5);
            else
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            // use [0 50]  > 50 remove outlies 
            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
            {
                count--;
                continue;
            }
        }
        //5.5.4当雷达线束不在以上范围，报错并跳出
        else
        {
            printf("wrong scan number\n");
            ROS_BREAK();
        }
        //printf("angle %f scanID %d \n", angle, scanID);
    //5.6
        //5.6.1计算该点的角度
        float ori = -atan2(point.y, point.x);
        //5.6.2当扫描没有过半，针对一些特殊角度重新计算ori
        if (!halfPassed)
        { 
            //2.1防止一些特殊情况
            if (ori < startOri - M_PI / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > startOri + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }
            //2.2当点过半，置当前扫描状态halfPassed为True
            if (ori - startOri > M_PI)
            {
                halfPassed = true;
            }
        }
        //5.6.3当扫描过半,针对一些特殊角度重新计算ori
        else
        {
            ori += 2 * M_PI;
            if (ori < endOri - M_PI * 3 / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > endOri + M_PI / 2)
            {
                ori -= 2 * M_PI;
            }
        }
        //5.6.4获取时间
        float relTime = (ori - startOri) / (endOri - startOri);//relTime 是一个0~1之间的小数，代表占用扫描时间的比例，乘以扫描时间得到真实扫描时刻，
        point.intensity = scanID + scanPeriod * relTime;//intensity分两部分，正数部分是：线束scanID。小数部分是，在这个扫描周期里，这个激光是第几s发出的。
        laserCloudScans[scanID].push_back(point);//将点加入点云
    }

//6)重新组织点云
    //6.1重新获取点云时间
    cloudSize = count;
    printf("points size %d \n", cloudSize);
    //6.2新建点云
    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    //6.3遍历每一个激光束，将点云向量里的元素并入到一个点云里，并求出每一列点云的索引。
    for (int i = 0; i < N_SCANS; i++)  
    { 
        //6.3.1求出第i列点云开始的索引
        scanStartInd[i] = laserCloud->size() + 5;
        //6.3.2将当前激光束的这一列点云加入laserCloud中
        *laserCloud += laserCloudScans[i];
        //6.3.3求出第i列点云结束的索引                
        scanEndInd[i] = laserCloud->size() - 6;  //scanEndInd和scanStartInd是对应的。比如第3列开始时候是170点 第3列共8个点，那么scanEndInd会是178，第4列的scanStartInd就是179
    }
    //6.4打印时间
    printf("prepare time %f \n", t_prepare.toc());
    
//7)计算曲率
    //7.1遍历点云
    for (int i = 5; i < cloudSize - 5; i++)
    {
    //7.2用同一列的激光线上10个点计算曲率。先对各个方向求一个中间变量
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;
    //7.3获取曲率
        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
    //7.4该点的其他信息
        cloudSortInd[i] = i;//点的索引  （后面分类会用到）
        cloudNeighborPicked[i] = 0;//该点是否为特征点
        cloudLabel[i] = 0;//点的类别
    }
    
    TicToc t_pts;
//8)
    //8.1定义四种特征点
    pcl::PointCloud<PointType> cornerPointsSharp;    //点特征（曲率特别大）
    pcl::PointCloud<PointType> cornerPointsLessSharp;//点特征（曲率大）
    pcl::PointCloud<PointType> surfPointsFlat;       //面特征（曲率特别小）
    pcl::PointCloud<PointType> surfPointsLessFlat;   //面特征（曲率小）
    
    float t_q_sort = 0;

    //8.2根据曲率提取特征点，并对特征点分类
    for (int i = 0; i < N_SCANS; i++)
    {
        
        //8.2.1当该激光束的扫描点小于6,直接跳出。（因为无法分成6份）
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        //8.2.2定义一个点云，用以存储特征点
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
        //8.2.3进入循环，遍历每一个扇区
        for (int j = 0; j < 6; j++)
        {
            //3.1将每个scan等分成6等份，也就是6个扇区。对曲率进行排序
                //3.1.1计算subscan的index
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; // subscan的起始index       j=0,sp=scanStartInd[i]; j=1, sp=scanStartInd[i]+第i线的scan点数的1/6
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;//subscan的结束index
                //3.1.2根据曲率由小到大对subscan的点进行sort;  对每一线都得到了6个曲率由小到大的点集
            TicToc t_tmp;
            std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp);//sort（begin,end,cmp）begin为被sort的数组第一个元素的指针。end为待sort的数组最后一个元素的下一个指针 cmp为排序准则
                //3.1.3计算每个扇区排序时间总和
            t_q_sort += t_tmp.toc();
                
            //3.2提取点特征点 按照曲率从高到低遍历扇区中的每一个点
            int largestPickedNum = 0;
            for (int k = ep; k >= sp; k--)
            {
                //3.2.1取出点在整个点云的索引
                int ind = cloudSortInd[k]; //cloudSortInd进过sort不再是按照索引顺序排列，而是曲率大小
                //3.2.2当点还没经过特征判断，并且曲率大于0.1
                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] > 0.1)
                {
                //3.2.3对特征点分类
                    largestPickedNum++;
                    //3.1对于曲率前2的点进行操作
                    if (largestPickedNum <= 2)//第一次进入循环的点和第二次进入循环的点会进入这条if语句。
                    {                        
                        //3.1.1将点的label置为2
                        cloudLabel[ind] = 2;//label2代表曲率特别大的点
                        //3.1.2将点放入点特征（曲率特大组）
                        cornerPointsSharp.push_back(laserCloud->points[ind]);
                        //3.1.3将点放入点特征  因为曲率特大也是属于曲率大，所以也会被放到cornerPointsLessSharp
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    //3.2对于曲率前20的点进行操作
                    else if (largestPickedNum <= 20)
                    {                        
                        //3.2.1将点的label置为1
                        cloudLabel[ind] = 1; //label1代表曲率大
                        //3.2.2将点放入特征点
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    //3.3特征点数量已经充足，跳出循环
                    else
                    {
                        break;
                    }
                    //3.4该点已经过特征判断，状态置为1
                    cloudNeighborPicked[ind] = 1; 
                //3.2.4对于特征点周围（或者说前后）距离平方<=0.05的点标记为已经过特征筛选。  防止特征点聚集，使得特征点在每个方向上尽量分布均匀
                    //4.1对前5个点进行判断
                    for (int l = 1; l <= 5; l++)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    //4.2对后5个点进行判断
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }


            //3.3提取面特征 按照曲率从低到高遍历扇区中每一个点
            int smallestPickedNum = 0;
                //3.3.1
            for (int k = sp; k <= ep; k++)
            {
                    //1.1取出点在整个点云的索引
                int ind = cloudSortInd[k];
                    //1.2当点还没经过特征判断，并且曲率小于0.3
                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] < 0.3)
                {
                    //1.3对特征点分类
                        //3.1将点的label置为-1
                    cloudLabel[ind] = -1; 
                        //3.2将点放入面特征（曲率特别小）
                    surfPointsFlat.push_back(laserCloud->points[ind]);
                    
                    smallestPickedNum++;
                        //3.3当特征点数量充足，跳出循环
                    if (smallestPickedNum >= 4)
                    { 
                        break;
                    }
                        //3.4该点已经过特征判断，状态置为1
                    cloudNeighborPicked[ind] = 1;

                    //1.4对于特征点周围（或者说前后）距离平方<=0.05的点标记为已经过特征筛选。  防止特征点聚集，使得特征点在每个方向上尽量分布均匀                 
                        //4.1对前5个点进行判断
                    for (int l = 1; l <= 5; l++)
                    { 
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                        //4.2对后5个点进行判断
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }
                //3.3.2剩余点归于面特征点（曲率小）
            for (int k = sp; k <= ep; k++)
            {
                if (cloudLabel[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }
        }
        //8.2.4对面特征点（曲率小）进行降采样
            //4.1降采样
        pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
        pcl::VoxelGrid<PointType> downSizeFilter;
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilter.filter(surfPointsLessFlatScanDS);
            //4.2将点云汇聚到一起
        surfPointsLessFlat += surfPointsLessFlatScanDS;
    }
    //8.3打印消耗时间
    printf("sort q time %f \n", t_q_sort);
    printf("seperate points time %f \n", t_pts.toc());

    //8.4发布四种特征点和点云
        //8.4.1发布点云
            //1.1创建发布消息实例
    sensor_msgs::PointCloud2 laserCloudOutMsg;
            //1.2将点云转为发布消息类型
    pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
            //1.3封装消息的header
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
    laserCloudOutMsg.header.frame_id = "/camera_init";
            //1.4发布
    pubLaserCloud.publish(laserCloudOutMsg);

        //8.4.2发布点特征（曲率特别大）
    sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsSharp.publish(cornerPointsSharpMsg);

        //8.4.3发布点特征（曲率大）
    sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
    cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsLessSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

        //8.4.4发布面特征（曲率特别小）
    sensor_msgs::PointCloud2 surfPointsFlat2;
    pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
    surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsFlat2.header.frame_id = "/camera_init";
    pubSurfPointsFlat.publish(surfPointsFlat2);

        //8.4.5发布面特征（曲率小）
    sensor_msgs::PointCloud2 surfPointsLessFlat2;
    pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
    surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsLessFlat2.header.frame_id = "/camera_init";
    pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

        //8.4,6当需要发布按scan线发布时，发布点云
    if(PUB_EACH_LINE)
    {
        for(int i = 0; i< N_SCANS; i++)
        {
            sensor_msgs::PointCloud2 scanMsg;
            pcl::toROSMsg(laserCloudScans[i], scanMsg);
            scanMsg.header.stamp = laserCloudMsg->header.stamp;
            scanMsg.header.frame_id = "/camera_init";
            pubEachScan[i].publish(scanMsg);
        }
    }
    //8.5打印消耗时间
    printf("scan registration time %f ms *************\n", t_whole.toc());
    if(t_whole.toc() > 100)
        ROS_WARN("scan registration process over 100ms");
}

int main(int argc, char **argv)
{
//1)ROS初始化
    ros::init(argc, argv, "scanRegistration");
    ros::NodeHandle nh;
//2)加载配置文件
    //2.1从配置参数中 读取 scan_line 参数, 多少线的激光雷达  在launch文件中配置的
    nh.param<int>("scan_line", N_SCANS, 16);
    //2.2//从配置参数中 读取 minimum_range 参数, 最小有效距离  在launch文件中配置的   踢出雷达上的载体出现在视野里的影响
    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);
    //2.3打印雷达线数
    printf("scan line number %d \n", N_SCANS);
//3)算法仅仅支持16、32、64线雷达
    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }
//4)创建订阅
    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/kitti/velo/pointcloud", 100, laserCloudHandler);//订阅点云
    // ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);//订阅点云
//5)创建发布
    //5.1发布对象
    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);

    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);//点特征 曲率特别大

    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);//点特征 曲率大

    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);//面特征 曲率特别小

    pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);//面特征 曲率小

    pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);
    //5.2当需要发布每个scan时
    if(PUB_EACH_LINE)
    {
        for(int i = 0; i < N_SCANS; i++)
        {
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
            pubEachScan.push_back(tmp);
        }
    }
//6)执行回调
    ros::spin();

    return 0;
}
