import numpy as np
from scipy.spatial.transform import Rotation as R

def convert_camera_to_robot_quaternion(yaw_cam, pitch_cam, roll_cam):
    """
    Converts cleaned and axis-swapped camera Euler angles to a robot-frame quaternion.
    Applies:
    1. +90° to yaw and roll (camera mounting compensation)
    2. Angle clamping between -5° and 5° → 0°
    3. Swapping roll and pitch due to rotated camera axes
    4. Conversion to robot-frame quaternion

    Returns:
        quat_robot: Quaternion [x, y, z, w]
    """
    def clamp_small(angle):
        return 0.0 if -10.0 <= angle <= 10.0 else angle
    
    def clamp_180(angle):
        return 180.0 if 170.0 <= angle <= 190 else angle

    tilted_left = False
    tilted_right = False
    flat = False

    # Step 1: Add 90° to yaw and roll
    #yaw_cam += 90
    roll_cam += 35

    # Step 2: Clamp small values to 0
    yaw_cam = clamp_small(yaw_cam)
    pitch_cam = clamp_small(pitch_cam)
    roll_cam = clamp_small(roll_cam)

    print(f"[DEBUG] Adjusted angles before swap (deg): Yaw={yaw_cam}, Pitch={pitch_cam}, Roll={roll_cam}")

    # Step 3: Swap roll and pitch due to camera mounting
    pitch_robot = (clamp_180(pitch_cam))
    roll_robot = clamp_180(roll_cam)
    yaw_robot = clamp_180(yaw_cam)

    #pitch2 = pitch_robot
    #pitch_robot = -(90 - pitch_robot)
    if pitch_robot != 0:
        pitch_robot = -90 - pitch_robot
    else:
        pitch_robot = -(yaw_robot)
    yaw_robot = yaw_robot
    roll_robot = roll_robot
    print(f"[DEBUG] Robot-frame Euler angles (deg): Yaw={yaw_robot}, Pitch={pitch_robot}, Roll={roll_robot}")

    # Step 4: Convert to quaternion using ZYX (yaw → pitch → roll)
    #r_robot = R.from_euler('zyx', [yaw_robot, pitch_robot, roll_robot], degrees=True)
    r_robot = R.from_euler('xyz', [roll_robot, pitch_robot, yaw_robot], degrees=True)
    quat_robot = r_robot.as_quat()

    print(f"[INFO] Robot-frame quaternion: {quat_robot}")
    if yaw_robot == 0:
        if pitch_robot > -90:
            tilted_right = True
        else:
            tilted_left = True
    else:
        flat = True
        quat_robot = [0.707106781,0,0.707106781,0]

    return quat_robot, tilted_right, tilted_left, flat