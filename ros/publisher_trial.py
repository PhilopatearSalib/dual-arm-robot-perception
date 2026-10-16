#!/usr/bin/env python
"""
Proof of concept for the perception -> motion-planning interface.

Publishes a MoveIt Grasp message on /grasp every 2 seconds. The grasp position is a predefined
value (the clean-pan station), standing in for a position that the detector would provide.
"""
import rospy
from moveit_msgs.msg import Grasp

rospy.init_node('classifier_node', anonymous=True)
classifier_pub = rospy.Publisher("/grasp", Grasp, queue_size=1)
grasps = Grasp()

grasps.grasp_pose.pose.position.x = 0.415
grasps.grasp_pose.pose.position.y = 0

while not rospy.is_shutdown():
    classifier_pub.publish(grasps)
    rospy.sleep(2)
