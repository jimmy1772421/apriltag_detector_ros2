#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class AprilTagDetector(Node):
    def __init__(self):
        super().__init__('apriltag_detector')

        #create cv bridge to convert ROS images to OpenCV format
        self.bridge = CvBridge()

        #setup aruco
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # Camera calibration (you'd need to calibrate your camera)
        self.camera_matrix = np.array([[800, 0, 320],
                                      [0, 800, 240], 
                                      [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4,1))  # Assuming no distortion
        self.marker_size = 0.05  # 5cm marker size


        #create a subscriber to the image topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        #publish output image
        self.publisher = self.create_publisher(
            Image,
            "/apriltag/detected_image",
            10
        )

        self.get_logger().info('AprilTag detector started, waiting for imaging')

    def image_callback(self, msg):
        try:
            #convert ros image to opencv image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            #convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)


            #detect apriltags
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.aruco_params
            )

            #check if found tag
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                # Get 3D pose
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.camera_matrix, self.dist_coeffs)
                
                for i in range(len(ids)):
                    cv2.aruco.drawAxis(cv_image, self.camera_matrix, 
                                    self.dist_coeffs, rvecs[i], tvecs[i], 0.03)
                    
                    # Log 3D position and rotation
                    pos = tvecs[i][0]
                    rot = rvecs[i][0]
                    self.get_logger().info(f'Tag {ids[i][0]}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]m')
                    
                    self.get_logger().info(f'Found {len(ids)} AprilTag: {ids.flatten()}')

            else:
                self.get_logger().info('No AprilTag found')

            #log
            #height, width = cv_image.shape[:2]
            #self.get_logger().info(f'Recieved image : {width}x{height}')

            #convert back to ros image
            output_msg = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
            self.publisher.publish(output_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
    rclpy.init()
    node = AprilTagDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()