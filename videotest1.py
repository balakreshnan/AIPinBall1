#!/usr/bin/env python

import cv2
import time

if __name__ == '__main__' :

    # Start default camera
    # video = cv2.VideoCapture(0);
    video = cv2.VideoCapture('C:\\Users\\babal\\Downloads\\WIN_20220920_11_27_37_Pro.mp4')

    #video = cv2.VideoCapture('C:\\Users\\babal\\Downloads\\WIN_20220920_11_27_37_Pro.mp4', apiPreference=cv2.CAP_ANY, params=[
    ##cv2.CAP_PROP_FRAME_WIDTH, 1280,
    #cv2.CAP_PROP_FRAME_HEIGHT, 1024])
    video.set(cv2.CAP_PROP_FPS, 60.0)
    video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
    video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 2200)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 2000)

    if not video.isOpened():
        print("Cannot open camera")
        exit()

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.

    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    width = video.get(cv2.CAP_PROP_FRAME_WIDTH )
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT )
    fps =  video.get(cv2.CAP_PROP_FPS)

    # Number of frames to capture
    num_frames = 120;

    print("Capturing {0} frames".format(num_frames))

    # Start time
    start = time.time()

    # Grab a few frames
    for i in range(0, num_frames) :
        ret, frame = video.read()

    # End time
    end = time.time()

    # Time elapsed
    seconds = end - start
    print ("Time taken : {0} seconds".format(seconds))

    # Calculate frames per second
    fps  = num_frames / seconds
    print("Estimated frames per second : {0}".format(fps))

    #h, w, c = video.shape
    print('width:  ', width)
    print('height: ', height)
    print('fps:', fps)
    print('FPS ',video.get(cv2.CAP_PROP_FPS))
    print('Width ',video.get(cv2.CAP_PROP_FRAME_WIDTH))
    print('Height ',video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('Fourcc' ,video.get(cv2.CAP_PROP_FOURCC))
    print('Hue' ,video.get(cv2.CAP_PROP_HUE))
    print('RGB' ,video.get(cv2.CAP_PROP_CONVERT_RGB))
    print('Zoom: ' , video.get(cv2.CAP_PROP_ZOOM))
    print('Channel' , video.get(cv2.CAP_PROP_CHANNEL))
    print('Aspect ratio', video.get(cv2.CAP_PROP_SAR_NUM))
    print('Bit rate', video.get(cv2.CAP_PROP_BITRATE))
    # print('mode' , video.get(cv2.CAP_MODE_BGR))

    #sky = video[0:100, 0:200]
    # cv2.imshow('Video', video)
    while True:
        ret, frame = video.read()
        # (height, width) = frame.shape[:2]
        #sky = frame[0:100, 0:200]
        #cv2.imshow('Video', sky)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Video', frame)
        imS = cv2.resize(frame, (960, 540))                # Resize image
        cv2.imshow("output", imS)

        if cv2.waitKey(1) == 27:
            exit(0)

    # Release video
    video.release()