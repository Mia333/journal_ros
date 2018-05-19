# journal_ros

### ROS install tutorial ###
https://blog.csdn.net/softimite_zifeng/article/details/78632211
https://www.cnblogs.com/liu-fa/p/5779206.html
查看可用的package
$　apt-cache search ros-kinetic

初始化rosdep
$　sudo rosdep init 
$ rosdep update 

初始化环境变量
$　echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
$　source ~/.bashrc

8. 安装building package的依赖
$　sudo apt-get install python-rosinstall python-rosinstall-generator python-wstool build-essential 

9. 测试ROS安装成功
1) 打开Termial，输入以下命令，初始化ROS环境：
$ roscore 

2) 打开新的Termial，输入以下命令，弹出一个小乌龟窗口：
$ rosrun turtlesim turtlesim_node 

** ros中很多的第三方插件的安装格式是：
$ sudo apt-get install ros-kinetic-...
例如：
$ sudo apt-get install ros-kinetic-turtlebot*
