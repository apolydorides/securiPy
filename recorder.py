import numpy as np
import cv2 as cv
import time

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    
# VARIABLES
prev_time = time.time()
# List of 8 consecutive frames stored for differencing and summing technique
frame_mat = []
# Difference of frame_mat and background frame
diff_mat = []
# counter for 50 initial frames used for background
captured_frames = 0
# stores the 50 frames
background_frames = []
# recording frame rate
frame_rate = 25

#------------------------------------------------------------------------------------------#

# FUNCTION DEFINITIONS
def get_background():
    # set the correct scope for these variables
    global captured_frames, prev_time, frame_rate, background_frames
    while (captured_frames < 50):
        # Captures frame-by-frame
        success, frame = cap.read()
        
        # if frame is read correctly ret is True
        if not success:
            print("Can't receive frame (stream end?). Exiting...")
            break
        
        # cannot guarantee any camera will support X fps
        # neither all backends (CAP_X) work with platforms
        # SO, manually using frames every 1/fps seconds
        time_elapsed = time.time() - prev_time
        
        if time_elapsed  > 1./frame_rate:
            prev_time = time.time()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            background_frames.append(gray.flatten())
            captured_frames = captured_frames + 1
    
    # once all 50 frames are captured, return the mean of them as the background of the camera
    return np.median(background_frames, axis=0)


#------------------------------------------------------------------------------------------#

# MAIN CODE
bg_frame = get_background()
    
while True:
    # Captures frame-by-frame
    success, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not success:
        print("Can't receive frame (stream end?). Exiting...")
        break
    
    time_elapsed = time.time() - prev_time
    
    if time_elapsed  > 1./frame_rate:
        prev_time = time.time()
        # OperationS on the frame come here
        resized = cv.resize(frame, (960, 540))
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # DIFFERENCING AND SUMMING TECHNIQUE
        new_arr = gray.flatten()
        if len(frame_mat) < 8:
            frame_mat.append(new_arr)
        else:
            frame_mat.insert(0, new_arr)
            frame_mat.pop(8)
            ret, diff_mat = cv.threshold(abs(frame_mat - bg_frame), 50, 255, cv.THRESH_BINARY)
            summed_diff = np.reshape(np.sum(diff_mat, axis=0), (1080, 1920))
            cv.imshow('summed_diff', summed_diff)
            # morphological transformation
            kernel = np.ones((3,3), np.uint8)
            closing = cv.morphologyEx(summed_diff, cv.MORPH_OPEN, kernel)
            opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
            cv.imshow('opening', opening)
            print(opening)
        
        
        # Display the resulting frame
        cv.imshow('frame', resized)
    
    if cv.waitKey(1) == ord('q'):
        break
    
# When everything is done, release the capture
cap.release()
cv.destroyAllWindows()

