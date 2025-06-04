# ABB IRB 120 Bin-Picking System (Python + RAPID)

This project integrates a 6D pose estimation pipeline in **Python** with **ABB RAPID** robot programming to perform automated 
cube picking using an **Intel RealSense camera** and an **ABB IRB 120 robot**. The system detects cubes, estimates their 6D pose,
transforms the coordinates to robot space, and communicates this information over a TCP socket to perform precise pick-and-place operations.

---

## Project Overview

- **Input**: RGB-D image from RealSense camera
- **Processing**:
  - On-click cube segmentation (SAM)
  - 2D centroid and corner detection
  - 6D pose estimation using `solvePnP`
  - Coordinate transformation to robot frame
- **Output**: Formatted position and quaternion sent via TCP socket to RAPID
- **Execution**: Robot receives pose and executes `MoveJ` to pick the cube

---
## File Structure (The files used for the project, others in the Git were tested but not used)

├── 45degreecamera/                        # Python vision + control system
│   ├── main.py                            # Main script: segmentation → pose → send
│   ├── realsense_utils.py                 # RealSense setup and frame alignment
│   ├── sam.py                             # Segment Anything Model for on-click masking
│   ├── find_centroid.py                   # Finds centroid of segmented mask
│   ├── solvePNP.py                        # Estimates 6D pose using PnP from corners
│   ├── transform.py                       # Converts camera to robot coordinates
│   ├── pose_transform.py                  # Builds pose matrix, extracts rotation
│   ├── cube.py                            # Cube class to hold position and orientation
│   ├── send_to_robot.py                   # TCP/IP client to send pose to robot
│   ├── requirements.txt                   # Python dependencies
│
├── 2. RobotStudio Code/                   # RAPID programs for ABB IRB 120
│   ├── MainModule.mod                     # Main RAPID logic (socket, receive, move)
│   ├── PickPlace.mod                      # Pick-and-place motion routines
│
└── README.md                              # Project overview and setup guide


## Running the Program
The python files are run through main.py, The RobotStudio code is synchronised to the IRC5 cotnroller to run 
via the FlexPendant

## Author
Nathan Di Pietro (19451710) Curtin University




