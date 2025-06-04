MODULE PickPlace
    !***********************************************************
    !
    ! Module:  PickPlace
    !
    ! Description:
    !   For pick and place of cubes
    !
    ! Original Code by 19784490 (2023)
    ! Revised by 20173005 (2024)
    ! Revised by 19451710 (2025
    !
    !***********************************************************
    
    
    !***********************************************************
    !
    ! Global Variables
    !
    !***********************************************************
     
    ! Tool Data
    PERS tooldata Suction_Gripper:=[TRUE,[[0,0,205],[1,0,0,0]],[1,[5.311,-5.76,184.806],[1,0,0,0],0,0,0]];
    
    ! Positions
    ! auto-generated from simulations
    CONST robtarget Home:=[[50,0,800],[0.707106781,0,0.707106781,0],[0,0,0,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Camera_Pick:=[[156.953224998,-169.221348489,739.397581873],[0.707106781,0,0.707106781,0],[0,-1,0,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Camera_Place:=[[45.460847582,225,680],[0.707106781,0,0.707106781,0],[-1,0,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget WallReadyPose:=[[550, -26.124, 730], [0.84, -0.25, 0.46, 0.14], [-1, 1, 1, 1], [9E+09, 9E+09, 9E+09, 9E+09, 9E+09, 9E+09]];

    
    CONST robtarget Cube1_Box:=[[575,-26.124,760.893],[0.707106781,0,0.707106781,0],[-1,-2,1,1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube1_Box_Above:=[[525,-26.124,760.893],[0.707106781,0,0.707106781,0],[-1,-2,1,1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube1_Box_Above_Above:=[[450,-26.124,760.893],[0.707106781,0,0.707106781,0],[-1,-2,1,1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube2_Box:=[[575,-102.593,527.055],[0.707106781,0,0.707106781,0],[-1,-1,0,1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube2_Box_Above:=[[525,-102.593,527.055],[0.707106781,0,0.707106781,0],[-1,-1,0,1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube2_Box_Above_Above:=[[450,-102.593,527.055],[0.707106781,0,0.707106781,0],[-1,-1,0,1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube3_Box:=[[575,11.371787342,626.22735512],[0.707106781,0,0.707106781,0],[1,-2,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube3_Box_Above:=[[525,11.371787342,626.227],[0.707106781,0,0.707106781,0],[1,-2,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube3_Box_Above_Above:=[[450,11.371787342,626.227],[0.707106781,0,0.707106781,0],[1,-2,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube4_Box:=[[575,-58.122787342,349.64635512],[0.707106781,0,0.707106781,0],[-1,-1,0,1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube4_Box_Above:=[[525,-58.123,349.646],[0.707106781,0,0.707106781,0],[-1,-1,0,1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube4_Box_Above_Above:=[[450,-58.123,349.646],[0.707106781,0,0.707106781,0],[-1,-1,0,1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube5_Box:=[[575,81.158787342,405.59135512],[0.707106781,0,0.707106781,0],[1,-2,0,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube5_Box_Above:=[[525,81.159,405.591],[0.707106781,0,0.707106781,0],[1,-2,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube5_Box_Above_Above:=[[450,81.159,405.591],[0.707106781,0,0.707106781,0],[1,-2,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube6_Box:=[[575,131.158787342,555.59135512],[0.707106781,0,0.707106781,0],[1,-2,0,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube6_Box_Above:=[[525,131.158787342,555.59135512],[0.707106781,0,0.707106781,0],[1,-2,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube6_Box_Above_Above:=[[450,131.158787342,555.59135512],[0.707106781,0,0.707106781,0],[1,-2,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube7_Box:=[[575,131.158787342,705.59135512],[0.707106781,0,0.707106781,0],[1,-2,0,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube7_Box_Above:=[[525,131.158787342,705.59135512],[0.707106781,0,0.707106781,0],[1,-2,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube7_Box_Above_Above:=[[450,131.158787342,705.59135512],[0.707106781,0,0.707106781,0],[1,-2,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube8_Box:=[[575,-102.593,727.055],[0.707106781,0,0.707106781,0],[1,-2,0,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube8_Box_Above:=[[525,-102.593,727.055],[0.707106781,0,0.707106781,0],[1,-2,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube8_Box_Above_Above:=[[450,-102.593,727.055],[0.707106781,0,0.707106781,0],[1,-2,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    
    CONST robtarget Cube1_Stack:=[[300,225,680],[0.707106781,0,0.707106781,0],[-2,1,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube1_Stack_Above:=[[250,225,680],[0.707106781,0,0.707106781,0],[-1,0,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube2_Stack:=[[250,225,680],[0.707106781,0,0.707106781,0],[-1,0,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube2_Stack_Above:=[[200,225,680],[0.707106781,0,0.707106781,0],[-1,0,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube3_Stack:=[[200,225,680],[0.707106781,0,0.707106781,0],[-1,0,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube3_Stack_Above:=[[150,225,680],[0.707106781,0,0.707106781,0],[-1,0,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    
    CONST robtarget Cube1_Pyramid:=[[305,325,680],[0.707106781,0,0.707106781,0],[-2,1,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube1_Pyramid_Above:=[[250,325,680],[0.707106781,0,0.707106781,0],[-1,0,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube2_Pyramid:=[[305,268.781,674.635],[0.707106781,0,0.707106781,0],[-2,1,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube2_Pyramid_Above:=[[250,268.781,674.635],[0.707106781,0,0.707106781,0],[-1,0,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube3_Pyramid:=[[305,267.659,730.521],[0.707106781,0,0.707106781,0],[-2,1,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube3_Pyramid_Above:=[[250,267.659,730.521],[0.707106781,0,0.707106781,0],[-1,0,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube4_Pyramid:=[[305,323.301,727.371],[0.707106781,0,0.707106781,0],[-2,1,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube4_Pyramid_Above:=[[250,323.301,727.371],[0.707106781,0,0.707106781,0],[-1,0,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube5_Pyramid:=[[255,298.908,704.708],[0.707106781,0,0.707106781,0],[-1,0,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Cube5_Pyramid_Above:=[[200,298.908,704.708],[0.707106781,0,0.707106781,0],[-1,0,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    
    CONST robtarget zPick_Limit_1:=[[575,14.721237546,780.519958444],[0.707106781,0,0.707106781,0],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget zPick_Limit_2:=[[575,-387.378246549,600.626374281],[0.707106781,0,0.707106781,0],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget zPick_Limit_3:=[[575,-468.510415628,134.317525302],[0.707106781,0,0.707106781,0],[-1,1,-2,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    
    VAR socketdev clientSocket;
    VAR socketdev serverSocket;
    VAR string receivedData;
    VAR num bytesReceived;
    VAR string client_ip;
    
    VAR string inputString := "255,123,64,0";
    VAR string tempValue;
    VAR num index := 1;
    VAR num position;
    VAR num start := 1;
    VAR num stringValue;
    VAR bool success;
    VAR bool ok;
    
    VAR num pickX;
    VAR num pickY;
    VAR num pickZ;
    VAR num pickRotation;
    VAR num qx;
    VAR num qy;
    VAR num qz;
    VAR num qw;
    
    VAR num placeX;
    VAR num placeY;
    VAR num placeZ;
    VAR num placeRotation;
    
    VAR num stackLayer;
    VAR num stackPosition;
    VAR num blockNumber;
    
    !***********************************************************
    !
    ! FINAL Procedures
    !
    !***********************************************************
    
    !***********************************************************
    !
    ! TEST Procedures
    ! Use the code below to test functionalities and gain understanding
    !
    !***********************************************************

    ! Move away from camera view for testing computer vision
    PROC moveToCameraPlace()
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Place,v1000,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 3;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Place,v1000,z100,Suction_Gripper\WObj:=wobj0;
    ENDPROC
    
    ! Go to cube positions for calibration
    PROC goToCubePositions()
        MoveJ Home,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        !MoveJ Camera_Pick,v100,z100,Suction_Gripper\WObj:=wobj0;
        !WaitTime 2;
        
        ! Cube 1
        ConfJ \Off;
        MoveJ Cube1_Box_Above_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube1_Box_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube1_Box,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 5;
        MoveJ Cube1_Box_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube1_Box_Above_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        ConfJ \On;
        MoveJ Home,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        
        ! Cube 2
        ConfJ \Off;
        MoveJ Cube2_Box_Above_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube2_Box_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube2_Box,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 5;
        MoveJ Cube2_Box_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube2_Box_Above_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        ConfJ \On;
        MoveJ Home,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        
        ! Cube 3
        ConfJ \Off;
        MoveJ Cube3_Box_Above_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube3_Box_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube3_Box,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 5;
        MoveJ Cube3_Box_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube3_Box_Above_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        ConfJ \On;
        MoveJ Home,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        
        ! Cube 4
        ConfJ \Off;
        MoveJ Cube4_Box_Above_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube4_Box_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube4_Box,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 5;
        MoveJ Cube4_Box_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube4_Box_Above_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        ConfJ \On;
        MoveJ Home,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        
        ! Cube 5
        ConfJ \Off;
        MoveJ Cube5_Box_Above_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube5_Box_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube5_Box,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 5;
        MoveJ Cube5_Box_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube5_Box_Above_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        ConfJ \On;
        MoveJ Home,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        
        ! Cube 6
        ConfJ \Off;
        MoveJ Cube6_Box_Above_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube6_Box_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube6_Box,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 5;
        MoveJ Cube6_Box_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube6_Box_Above_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        ConfJ \On;
        MoveJ Home,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        
        ! Cube 7
        ConfJ \Off;
        MoveJ Cube7_Box_Above_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube7_Box_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube7_Box,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 5;
        MoveJ Cube7_Box_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube7_Box_Above_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        ConfJ \On;
        MoveJ Home,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        
        ! Cube 8
        ConfJ \Off;
        MoveJ Cube8_Box_Above_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube8_Box_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube8_Box,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 5;
        MoveJ Cube8_Box_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube8_Box_Above_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        ConfJ \On;
        MoveJ Home,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
    ENDPROC
    
    ! Go to the position of one cube
    ! Change robtarget name
    PROC goToCube()
        MoveJ Home,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Camera_Pick,v100,z100,Suction_Gripper\WObj:=wobj0;
        ConfJ \Off;
        WaitTime 2;
        MoveJ Cube8_Box_Above_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube8_Box_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube8_Box,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 8;
        MoveJ Cube8_Box_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube8_Box_Above_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        ConfJ \On;
        MoveJ Camera_Pick,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
    ENDPROC
    
    ! Wait for coordinates from Python script and auto pick and place blocks in stacks
    PROC autoStack()
        ! Initialize an empty string to store received data
        receivedData := "";
        
        ! Set blockNumber and stackLayer to 0
        blockNumber := 0;
        stackLayer:= 0;
        
        ! Move to Initial Position
        !MoveJ Camera_Place,v100,z50,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v100,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        
        ! Close sockets if already open
        SocketClose clientSocket;
        SocketClose serverSocket;
        
        ! Create a socket for communication (TCP/IP, Passive mode, port 5000)
        SocketCreate serverSocket;
        
        ! Bind clientSocket to computer IP address and port
        SocketBind serverSocket, "192.168.125.1", 5000;
    
        ! List for a connection from the Python client
        SocketListen serverSocket;
        
        ! If stocket connection available, accept connection
        TPWrite "Waiting for Connection...";
        SocketAccept serverSocket, clientSocket, \ClientAddress:=client_ip;
        TPWrite "Connection Successful!";
        
        WHILE TRUE DO

            ! Receive data from the client
            SocketReceive clientSocket \Str := receivedData;
            
            ! Delimit received data by , and split into different variables
            SplitStringToVariables;
            
            ! Increment block number
            blockNumber := blockNumber + 1;
    
            ! Print receiver and pick data
            TPWrite "Received data: "+receivedData;
            TPWrite "pickX: "+ NumToStr(pickX, 3);
            TPWrite "pickY: "+ NumToStr(pickY, 3);
            TPWrite "pickZ: "+ NumToStr(pickZ, 3);
            TPWrite "pickRotation: "+ NumToStr(pickRotation, 3);
            
            ! Reset index and start variables to 1
            index := 1;
            start := 1;
            
            ! Calculate the placement coordinates based on the current layer and position in the stack
            placeX := 680 - stackLayer * 100;
            placeY := 325;
            placeZ := 305 - (blockNumber - 1) MOD 3 * 50;
            
            ! Print place data
            TPWrite "placeX: "+ NumToStr(placeX, 3);
            TPWrite "placeY: "+ NumToStr(placeY, 3);
            TPWrite "placeZ: "+ NumToStr(placeZ, 3);
            
            ! Go to pick location
            MoveJ Home,v100,z100,Suction_Gripper\WObj:=wobj0;
            ConfJ \Off;
            MoveJ [[pickZ - 125,pickY,pickX],[0.707106781,0,0.707106781,0],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]],v500,z50,Suction_Gripper\WObj:=wobj0;
            MoveJ [[pickZ - 50,pickY,pickX],[0.707106781,0,0.707106781,0],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]],v500,z50,Suction_Gripper\WObj:=wobj0;
            MoveJ [[pickZ,pickY,pickX],[0.707106781,0,0.707106781,0],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]],v100,fine,Suction_Gripper\WObj:=wobj0;
            WaitTime 2;
            
            ! Turn end effector on
            SetDO Output1, 1;
            WaitTime 1;
            
            ! Go home
            MoveJ [[pickZ - 50,pickY,pickX],[0.707106781,0,0.707106781,0],[-2,1,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]],v100,z50,Suction_Gripper\WObj:=wobj0;
            MoveJ [[pickZ - 125,pickY,pickX],[0.707106781,0,0.707106781,0],[-2,1,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]],v500,z50,Suction_Gripper\WObj:=wobj0;
            ConfJ \On;
            MoveJ Home,v500,z50,Suction_Gripper\WObj:=wobj0;
            WaitTime 1;
            
            ! Go to place location
            MoveJ Camera_Place,v500,z50,Suction_Gripper\WObj:=wobj0;
            MoveJ [[placeZ - 50,placeY,placeX],[0.707106781,0,0.707106781,0],[-2,1,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]],v500,z50,Suction_Gripper\WObj:=wobj0;
            WaitTime 1;
            
            ! Rotate block
            WaitTime 1;
            MoveL RelTool ([[placeZ - 50,placeY,placeX],[0.707106781,0,0.707106781,0],[-2,1,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]], 0, 0, 0 \Rz:= pickRotation),v100,fine,Suction_Gripper\WObj:=wobj0;
            WaitTime 1;
            
            ! Place block
            MoveL RelTool ([[placeZ,placeY,placeX],[0.707106781,0,0.707106781,0],[-2,1,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]], 0, 0, 0 \Rz:= pickRotation),v100,fine,Suction_Gripper\WObj:=wobj0;
            WaitTime 1;
            
            ! Turn end effector off
            SetDO Output1, 0;
            WaitTime 1;
            
            ! Go home
            MoveJ [[placeZ - 50,placeY,placeX],[0.707106781,0,0.707106781,0],[-2,1,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]],v100,z50,Suction_Gripper\WObj:=wobj0;
            MoveJ Camera_Place,v500,z50,Suction_Gripper\WObj:=wobj0;
            WaitTime 2;

            ! Increment stackLayer so that the max stack height is 3
            IF blockNumber MOD 3 = 0 THEN
                stackLayer := stackLayer + 1;
            ENDIF
            
        ENDWHILE
        ERROR
            IF ERRNO=ERR_SOCK_CLOSED THEN
                ! Close sockets
                SocketClose clientSocket;
                SocketClose serverSocket;
                RETURN;
            ELSE
                !No error handling
            ENDIF
        UNDO
            ! Close the socket
            SocketClose clientSocket;
            SocketClose serverSocket;
    ENDPROC
    
    PROC autoPyramid()
        ! Initialize an empty string to store received data
        receivedData := "";
        
        ! Set blockNumber and stackLayer to 0
        blockNumber := 0;
        stackLayer:= 0;
        
        ! Move to Initial Position
        !MoveJ Camera_Place,v100,z50,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v500,z50,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        
        ! Close sockets if already open
        SocketClose clientSocket;
        SocketClose serverSocket;
        
        ! Create a socket for communication (TCP/IP, Passive mode, port 5000)
        SocketCreate serverSocket;
        
        ! Bind clientSocket to computer IP address and port
        SocketBind serverSocket, "192.168.125.1", 5000;
    
        ! List for a connection from the Python client
        SocketListen serverSocket;
        
        ! If stocket connection available, accept connection
        TPWrite "Waiting for Connection...";
        SocketAccept serverSocket, clientSocket, \ClientAddress:=client_ip;
        TPWrite "Connection Successful!";
        
        WHILE TRUE DO

            ! Receive data from the client
            SocketReceive clientSocket \Str := receivedData;
            
            ! Delimit received data by , and split into different variables
            SplitStringToVariables;
            
            ! Increment block number
            blockNumber := blockNumber + 1;
    
            ! Print receiver and pick data
            TPWrite "Received data: "+receivedData;
            TPWrite "pickX: "+ NumToStr(pickX, 3);
            TPWrite "pickY: "+ NumToStr(pickY, 3);
            TPWrite "pickZ: "+ NumToStr(pickZ, 3);
            !TPWrite "pickRotation: "+ NumToStr(pickRotation, 3);
            TPWrite "qx: "+ NumToStr(qx, 3);
            TPWrite "qy: "+ NumToStr(qy, 3);
            TPWrite "qz: "+ NumToStr(qz, 3);
            TPWrite "qw: "+ NumToStr(qw, 3);
            
            ! Reset index and start variables to 1
            index := 1;
            start := 1;
            
            ! Calculate the placement coordinates based on the pyramid structure
            IF blockNumber <= 3 THEN
                ! Base layer (3 blocks)
                placeX := 580 + (blockNumber - 1) * 65;
                placeY := 325;
                placeZ := 305;
            ELSEIF blockNumber <= 5 THEN
                ! Middle layer (2 blocks)
                placeX := 580 + ((blockNumber - 3) * 65 / 2 + (blockNumber - 4) * 65 / 2);
                placeY := 325;
                placeZ := 305 - 50;
            ELSE
                ! Top layer (1 block)
                placeX := 580 + 65;
                placeY := 325;
                placeZ := 305 - 2 * 50;
            ENDIF
            
            ! Print place data
            TPWrite "placeX: "+ NumToStr(placeX, 3);
            TPWrite "placeY: "+ NumToStr(placeY, 3);
            TPWrite "placeZ: "+ NumToStr(placeZ, 3);
            
            ! Go to pick location
            MoveJ Home,v500,z50,Suction_Gripper\WObj:=wobj0;
            ConfJ \Off;
            MoveJ [[pickZ - 125,pickY,pickX],[qx,qy,qz,qw],[-1, 1, 1, 1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]],v500,z50,Suction_Gripper\WObj:=wobj0;
            !MoveJ [[pickZ - 50,pickY,pickX],[0.707106781,0,0.707106781,0],[-2,1,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]],v500,z50,Suction_Gripper\WObj:=wobj0;
            !MoveJ [[pickZ,pickY,pickX],[0.707106781,0,0.707106781,0],[-2,1,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]],v100,fine,Suction_Gripper\WObj:=wobj0;
            !Added 3D rotation
            !ConfJ \Off;
            MoveJ [[pickZ,pickY,pickX],[qx,qy,qz,qw],[-1, 1, 1, 1],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]],v100,fine,Suction_Gripper\WObj:=wobj0;
            WaitTime 2;
            
            ! Turn end effector on
            SetDO Output1, 1;
            WaitTime 1;
            
            ! Go home
            MoveJ [[pickZ - 50,pickY,pickX],[0.707106781,0,0.707106781,0],[-2,1,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]],v100,z50,Suction_Gripper\WObj:=wobj0;
            MoveJ [[pickZ - 125,pickY,pickX],[0.707106781,0,0.707106781,0],[-2,1,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]],v500,z50,Suction_Gripper\WObj:=wobj0;
            ConfJ \On;
            MoveJ Home,v500,z50,Suction_Gripper\WObj:=wobj0;
            WaitTime 1;
            
            ! Go to place location
            MoveJ Camera_Place,v500,z50,Suction_Gripper\WObj:=wobj0;
            MoveJ [[placeZ - 50,placeY,placeX],[0.707106781,0,0.707106781,0],[-2,1,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]],v500,z50,Suction_Gripper\WObj:=wobj0;
            
            ! Rotate block
            WaitTime 1;
            MoveL RelTool ([[placeZ - 50,placeY,placeX],[0.707106781,0,0.707106781,0],[-2,1,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]], 0, 0, 0 \Rz:= pickRotation),v100,fine,Suction_Gripper\WObj:=wobj0;
            WaitTime 1;
            
            ! Place block
            MoveL RelTool ([[placeZ,placeY,placeX],[0.707106781,0,0.707106781,0],[-2,1,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]], 0, 0, 0 \Rz:= pickRotation),v100,fine,Suction_Gripper\WObj:=wobj0;
            WaitTime 1;
            
            ! Turn end effector off
            SetDO Output1, 0;
            WaitTime 1;
            
            ! Go home
            MoveJ [[placeZ - 50,placeY,placeX],[0.707106781,0,0.707106781,0],[-2,1,-1,4],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]],v100,z50,Suction_Gripper\WObj:=wobj0;
            MoveJ Camera_Place,v500,z50,Suction_Gripper\WObj:=wobj0;
            WaitTime 2;
            
            
        ENDWHILE
        ERROR
            IF ERRNO=ERR_SOCK_CLOSED THEN
                ! Close sockets
                SocketClose clientSocket;
                SocketClose serverSocket;
                RETURN;
            ELSE
                !No error handling
            ENDIF
        UNDO
            ! Close the socket
            SocketClose clientSocket;
            SocketClose serverSocket;
    ENDPROC
    
    ! Assign values to variables
    FUNC bool AssignValue(num index, num value)
        success := FALSE;
        IF index = 1 THEN
            pickX := value;
        ELSEIF index = 2 THEN
            pickY := value;
        ELSEIF index = 3 THEN
            pickZ := value;
        ELSEIF index = 4 THEN
            !pickRotation := value;
            qx := value;
        ELSEIF index = 5 THEN
            qy := value;
        ELSEIF index = 6 THEN
            qz := value;
        ELSEIF index = 7 THEN
            qw := value;
        ENDIF
        RETURN success;
    ENDFUNC
    
    ! Split string by comma
     PROC SplitStringToVariables()
        WHILE index <= 7 DO
            ! Find the position of the next comma
            position := StrFind(receivedData, start, ",");
            
            IF position = 0 THEN
                ! If no comma found, get the remaining string part
                tempValue := StrPart(receivedData, start, StrLen(receivedData) - start + 1);
                ok := StrToVal(tempValue, stringValue);
                success := AssignValue(index, stringValue);
                EXIT;
            ELSE
                ! Get the substring up to the comma
                tempValue := StrPart(receivedData, start, position - start);
                ok := StrToVal(tempValue, stringValue);
                success := AssignValue(index, stringValue);
                start := position + 1;
            ENDIF
    
            index := index + 1;
        ENDWHILE
    ENDPROC
    
    PROC Testing()
        ConfJ \On;
        MoveJ Home,v100,z100,Suction_Gripper\WObj:=wobj0;
        
        WaitTime 1;
        ConfJ \Off;
        MoveJ WallReadyPose,v500,z50,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
    ENDPROC
    PROC pulseEndEffectorA()
        MoveJ Cube1_Box,v100,fine,Suction_Gripper\WObj:=wobj0;
        PulseDO Attach_Cube_1;
        WaitTime 1;
        MoveL Cube1_Box_Above,v100,fine,Suction_Gripper\WObj:=wobj0;
        PulseDO Detach_Cube_1;
        WaitTime 1;
    ENDPROC
    PROC pulseEndEffectorB()
        MoveJ Cube1_Stack,v500,fine,Suction_Gripper\WObj:=wobj0;
        PulseDO Attach_Cube_1;
        WaitTime 1;
        MoveL Cube1_Stack_Above,v500,fine,Suction_Gripper\WObj:=wobj0;
        PulseDO Detach_Cube_1;
        WaitTime 1;
    ENDPROC
    PROC pickPlaceCube1()
        MoveJ Home,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube1_Box_Above,v100,fine,Suction_Gripper\WObj:=wobj0;
        MoveL Cube1_Box,v100,fine,Suction_Gripper\WObj:=wobj0;
        PulseDO Attach_Cube_1;
        WaitTime 1;
        MoveL Cube1_Box_Above,v100,fine,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube1_Stack_Above,v100,fine,Suction_Gripper\WObj:=wobj0;
        MoveL Cube1_Stack,v100,fine,Suction_Gripper\WObj:=wobj0;
        PulseDO Detach_Cube_1;
        WaitTime 1;
        MoveL Cube1_Stack_Above,v100,fine,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v100,z100,Suction_Gripper\WObj:=wobj0;
    ENDPROC
    PROC pickPlaceCube1NoGripper()
        MoveJ Home,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube1_Box_Above,v100,fine,Suction_Gripper\WObj:=wobj0;
        MoveL Cube1_Box,v100,fine,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        MoveL Cube1_Box_Above,v100,fine,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube1_Stack_Above,v100,fine,Suction_Gripper\WObj:=wobj0;
        MoveL Cube1_Stack,v100,fine,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        MoveL Cube1_Stack_Above,v100,fine,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v100,z100,Suction_Gripper\WObj:=wobj0;
    ENDPROC
    PROC testThreeCubes()
        MoveJ Home,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Pick,v500,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        MoveJ Cube1_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube1_Box,v500,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube1_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Place,v500,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        MoveJ Cube1_Stack_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Cube1_Stack,v50,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveL Cube1_Stack_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Pick,v500,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        MoveJ Cube2_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube2_Box,v500,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube2_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Place,v500,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        MoveJ Cube2_Stack_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Cube2_Stack,v50,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveL Cube2_Stack_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Pick,v500,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        MoveJ Cube3_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube3_Box,v500,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveJ Cube3_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Place,v500,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        MoveJ Cube3_Stack_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Cube3_Stack,v50,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 2;
        MoveL Cube3_Stack_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v500,z100,Suction_Gripper\WObj:=wobj0;
    ENDPROC
    PROC manualStack()
        MoveJ Home,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Pick,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube1_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube1_Box,v500,z100,Suction_Gripper\WObj:=wobj0;
        gripperOn;
        WaitTime 2;
        MoveJ Cube1_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Place,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube1_Stack_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Cube1_Stack,v50,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        gripperOff;
        WaitTime 1;
        MoveL Cube1_Stack_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Pick,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube2_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube2_Box,v500,z100,Suction_Gripper\WObj:=wobj0;
        gripperOn;
        WaitTime 2;
        MoveJ Cube2_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Place,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube2_Stack_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Cube2_Stack,v50,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        gripperOff;
        WaitTime 1;
        MoveL Cube2_Stack_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Pick,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube3_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube3_Box,v500,z100,Suction_Gripper\WObj:=wobj0;
        gripperOn;
        WaitTime 2;
        MoveJ Cube3_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Place,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Cube3_Stack_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Cube3_Stack,v50,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        gripperOff;
        WaitTime 1;
        MoveL Cube3_Stack_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v500,z100,Suction_Gripper\WObj:=wobj0;
    ENDPROC
    PROC manualPyramid()
        MoveJ Home,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Pick,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube1_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube1_Box,v500,z100,Suction_Gripper\WObj:=wobj0;
        gripperOn;
        WaitTime 2;
        MoveJ Cube1_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Place,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube1_Pyramid_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Cube1_Pyramid,v50,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        gripperOff;
        WaitTime 1;
        MoveL Cube1_Pyramid_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Pick,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube2_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube2_Box,v500,z100,Suction_Gripper\WObj:=wobj0;
        gripperOn;
        WaitTime 2;
        MoveJ Cube2_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Place,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube2_Pyramid_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Cube2_Pyramid,v50,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        gripperOff;
        WaitTime 1;
        MoveL Cube2_Pyramid_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Pick,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube3_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube3_Box,v500,z100,Suction_Gripper\WObj:=wobj0;
        gripperOn;
        WaitTime 2;
        MoveJ Cube3_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Place,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube3_Pyramid_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Cube3_Pyramid,v50,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        gripperOff;
        WaitTime 1;
        MoveL Cube3_Pyramid_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Pick,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube4_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube4_Box,v500,z100,Suction_Gripper\WObj:=wobj0;
        gripperOn;
        WaitTime 2;
        MoveJ Cube4_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Place,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube4_Pyramid_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Cube4_Pyramid,v50,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        gripperOff;
        WaitTime 1;
        MoveL Cube4_Pyramid_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Pick,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube5_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube5_Box,v500,z100,Suction_Gripper\WObj:=wobj0;
        gripperOn;
        WaitTime 2;
        MoveJ Cube5_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Place,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube5_Pyramid_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Cube5_Pyramid,v50,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        gripperOff;
        WaitTime 1;
        MoveL Cube5_Pyramid_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v500,z100,Suction_Gripper\WObj:=wobj0;
    ENDPROC
    PROC manualTestNew()
        MoveJ Home,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Pick,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube4_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube4_Box,v500,z100,Suction_Gripper\WObj:=wobj0;
        gripperOn;
        WaitTime 2;
        MoveJ Cube4_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Camera_Pick,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Place,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube4_Pyramid_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Cube4_Pyramid,v50,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        gripperOff;
        WaitTime 1;
        MoveL Cube4_Pyramid_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Pick,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube5_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube5_Box,v500,z100,Suction_Gripper\WObj:=wobj0;
        gripperOn;
        WaitTime 2;
        MoveJ Cube5_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Camera_Pick,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Place,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube5_Pyramid_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Cube5_Pyramid,v50,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        gripperOff;
        WaitTime 1;
        MoveL Cube5_Pyramid_Above,v100,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v500,z100,Suction_Gripper\WObj:=wobj0;
    ENDPROC
    PROC manualPickTest()
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Pick,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube1_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube1_Box,v500,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        MoveJ Cube1_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Pick,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube2_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube2_Box,v500,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        MoveJ Cube2_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Pick,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube3_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube3_Box,v500,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        MoveJ Cube3_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Pick,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube4_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube4_Box,v500,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        MoveJ Cube4_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL Camera_Pick,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube5_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Cube5_Box,v500,z100,Suction_Gripper\WObj:=wobj0;
        WaitTime 1;
        MoveJ Cube5_Box_Above,v500,z100,Suction_Gripper\WObj:=wobj0;
        MoveJ Home,v1000,z100,Suction_Gripper\WObj:=wobj0;
    ENDPROC
    PROC limitTest()
        MoveL zPick_Limit_1,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL zPick_Limit_2,v1000,z100,Suction_Gripper\WObj:=wobj0;
        MoveL zPick_Limit_3,v1000,z100,Suction_Gripper\WObj:=wobj0;
    ENDPROC
        
ENDMODULE