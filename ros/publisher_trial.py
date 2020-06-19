#! /usr/bin/env python
import rospy
from std_msgs.msg import Float64MultiArray , Header , Int32
from moveit_msgs.msg import MoveItErrorCodes, Grasp, PlaceLocation
from grasping_msgs.msg import FindGraspableObjectsAction
import numpy as np

rospy.init_node('classifier_node', anonymous=True)
classifier_pub = rospy.Publisher("/grasp", Grasp, queue_size=1)
grasps = Grasp()


grasps.grasp_pose.pose.position.x = 0.415
grasps.grasp_pose.pose.position.y = 0


while not rospy.is_shutdown():
    classifier_pub.publish(grasps)
	rospy.sleep(2)
