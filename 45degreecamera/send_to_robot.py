# robot_sender.py

import socket
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

def format_pose_message_from_components(x, y, z, quat):
    """
    Format a pose message from position and quaternion array.

    Args:
        x, y, z (float): Position in mm
        quat (array-like): Quaternion [qx, qy, qz, qw]

    Returns:
        str: Formatted pose message string for robot
    """
    qx, qy, qz, qw = quat  # Unpack quaternion components
    msg = f"{x:.2f},{y:.2f},{z:.2f},{qx:.6f},{qy:.6f},{qz:.6f},{qw:.6f}"
    return msg

def send_to_robot(cube_list, ip="192.168.125.1", port=5000, max_retries=None, retry_delay=1):
    """
    Sends all cube poses in a single message to the robot via socket.

    Args:
        cube_list (list): List of cube objects with .x, .y, .z, .quat
        ip (str): Robot IP address
        port (int): Robot port
        max_retries (int or None): Retry attempts
        retry_delay (float): Delay between retries in seconds
    """
    try:
        all_data_strings = []
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Connecting with IRC5...")
        s.connect((ip, port))

        for i, cube in enumerate(cube_list):
            x, y, z = cube.x, cube.y, cube.z
            quat = cube.quat  # Should be [qx, qy, qz, qw]

            # === Use your custom formatting function === #
            data_string = format_pose_message_from_components(x, y, z, quat)

            # # === Log for debug === #
            # print(f"Cube {i + 1} Pose:")
            # print(f"Position: {x:.2f}, {y:.2f}, {z:.2f}")
            # print(f"Quaternion: {quat}")
            # print(f"Formatted String: {data_string}")

            # all_data_strings.append(data_string)

        # === Join all messages into one semicolon-separated string === #
        # final_message = ";".join(all_data_strings)

            # === Send to robot === #
            attempt = 0
            while True:
                try:

                    print("Sending Combined Pose Data:")
                    print(data_string)
                    s.sendall(data_string.encode('utf-8'))

                    time.sleep(30)
                    break

                except Exception as e:
                    attempt += 1
                    print(f"[WARN] Attempt {attempt} failed: {e}")
                    if max_retries is not None and attempt >= max_retries:
                        print("[ERROR] Max retries reached. Giving up.")
                        break
                    print(f"[INFO] Retrying in {retry_delay} second(s)...")
                    time.sleep(retry_delay)

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

    finally:
        time.sleep(5)
        try:
            s.close()
            print("Connection closed.")
        except:
            print("Socket was never opened.")