MODULE MainModule
    !***********************************************************
    !
    ! Module:  MainModule
    !
    ! Description:
    !   Contains Main
    !   Edit to run desired procedures as required
    !   Comment out what is not needed
    !
    ! Original Code by 19784490 (2023)
    ! Revised by 20173005 (2024)
    ! Revised by 19451710 (2025)
    !
    !***********************************************************
    
    PROC main()
        TPWrite "Hello World, starting program..."; ! use TPWrite to show text on the FlexPendant
        
        ! ********** Final Procedures **********
        !autoStack;
        !autoPyramid;
        
        ! ********** Calibration **********
        ! Set all to 0
        !sync;
        
        
        ! ********** Testing **********
        !moveToCameraPlace;
        !goToCubePositions;
        !goToCube;
        Testing;
        
    
        ! ********** ********** **********
        ! Delay at the end of loop
        manualStack;
        WaitTime 2;
        
    ENDPROC
    
    
	PROC sync()
        ! Refer to 3HAC035728-001 Revision: V, Chapter 5.7
        ! 
        ! How to Use:
        ! 1. Calibrate manually through fine calibration using the motor release button
        ! 2. Run this procedure by changing the main procedure (i.e. comment out all other procedures exect "sync")
        !    You may need to manually reposition the joints to within their working range
        ! 3. Calibrate revolution counters - this will save the calibration state
        !
        MoveAbsJ [[0,0,0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]]\NoEOffs,v1000,fine,tool0;
	ENDPROC


ENDMODULE