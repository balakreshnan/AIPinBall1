import argparse
import pathlib
import numpy as np
import onnxruntime
import PIL.Image
import time
import pytesseract
from PIL import Image
import cv2
import onnx
from math import sqrt
import math

PROB_THRESHOLD = 0.40  # Minimum probably to show results.

print(" Onnx Runtime : " + onnxruntime.get_device())

#labels = ['ballfail','ballinendzone','flaphit','SteelBall','zone1','zone2']
# labels = ['ballfail','ballinendzone','ballinplunge','flaphit','SteelBall','zone1','zone2']
labels = ['BallinPlunger','EndZone','FailZone','GameOver','leftflap','RightFlap','SteelBall']

providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
]

class Model:
    def __init__(self, model_filepath):
        self.session = onnxruntime.InferenceSession(str(model_filepath), providers=providers)
        assert len(self.session.get_inputs()) == 1
        self.input_shape = self.session.get_inputs()[0].shape[2:]
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = {'tensor(float)': np.float32, 'tensor(float16)': np.float16}[self.session.get_inputs()[0].type]
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.is_bgr = False
        self.is_range255 = False
        onnx_model = onnx.load(model_filepath)
        for metadata in onnx_model.metadata_props:
            if metadata.key == 'Image.BitmapPixelFormat' and metadata.value == 'Bgr8':
                self.is_bgr = True
            elif metadata.key == 'Image.NominalPixelRange' and metadata.value == 'NominalRange_0_255':
                self.is_range255 = True

    def predict(self, image_filepath):
        #image = PIL.Image.open(image_filepath).resize(self.input_shape)
	    #height = image_filepath.shape[0]
	    #width = image_filepath.shape[1]
        #image_array = jetson.utils.cudaToNumpy(image_filepath)
        #image = PIL.Image.fromarray(image_array, 'RGB').resize(self.input_shape)
        # image = PIL.Image.frombuffer("RGBX", (720,1280), image_filepath).resize(self.input_shape)
        # image = image_filepath.resize(320,320)
        img = cv2.cvtColor(image_filepath, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(img)
        image = PIL.Image.fromarray(img, 'RGB').resize(self.input_shape)
        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        input_array = input_array.transpose((0, 3, 1, 2))  # => (N, C, H, W)
        if self.is_bgr:
            input_array = input_array[:, (2, 1, 0), :, :]
        if not self.is_range255:
            input_array = input_array / 255  # => Pixel values should be in range [0, 1]

        outputs = self.session.run(self.output_names, {self.input_name: input_array.astype(self.input_type)})
        return {name: outputs[i] for i, name in enumerate(self.output_names)}


    def print_outputs(outputs):
        assert set(outputs.keys()) == set(['detected_boxes', 'detected_classes', 'detected_scores'])
        for box, class_id, score in zip(outputs['detected_boxes'][0], outputs['detected_classes'][0], outputs['detected_scores'][0]):
            if score > PROB_THRESHOLD:
                # print("{class_id}")
                # print(f"Az Cog Label: {class_id}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")
            # print(f"Az Cog Label: {labels[class_id]}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")
                if (class_id >= 0 and class_id <= 3):
                    print(f"Az Cog Label: {labels[class_id]}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")

model_path = "steelball3/model.onnx"

model = Model(model_path)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def main():
    print("Hello World!")
    from PIL import Image
    #read the image
    #file_path = 'pinballframe8890.jpg'
    #img = Image.open('pinballframe8890.jpg')
    # define a video capture object
#vid = cv2.VideoCapture(0)

previousframe = None
prevx = 0
prevy = 0
prevw = 0
prevh = 0

vid = cv2.VideoCapture('C:\\Users\\babal\\Downloads\\WIN_20220920_11_27_37_Pro.mp4')
vid.set(cv2.CAP_PROP_FPS,90)
#ret = vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#ret = vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print('FPS ',vid.get(cv2.CAP_PROP_FPS))
print('Width ',vid.get(cv2.CAP_PROP_FRAME_WIDTH))
print('Height ',vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('Fourcc' ,vid.get(cv2.CAP_PROP_FOURCC))
print('Hue' ,vid.get(cv2.CAP_PROP_HUE))
print('RGB' ,vid.get(cv2.CAP_PROP_CONVERT_RGB))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, frame = vid.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(frame)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    start = time.process_time()
    outputs = model.predict(frame)
    assert set(outputs.keys()) == set(['detected_boxes', 'detected_classes', 'detected_scores'])
    for box, class_id, score in zip(outputs['detected_boxes'][0], outputs['detected_classes'][0], outputs['detected_scores'][0]):
	    if score > PROB_THRESHOLD:
	        if (class_id >= 0 and class_id <= 6):
                    print(f"Az Cog Label: {labels[class_id]}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")
                    x = np.int32(box[0] * frame.shape[1])
                    y = np.int32(box[1] * frame.shape[0])
                    w = np.int32(box[2] * frame.shape[1])
                    h = np.int32(box[3] * frame.shape[0])
                    point_one = (x,y)
                    point_two = (x + (w - x), y + (h - y))
	                # img1 = cv2.rectangle(jetson.utils.cudaToNumpy(img), point_one, point_two, color=(255,211,67), thickness=2)
                    cv2.rectangle(frame, point_one, point_two, color=(255,211,67), thickness=2)
                    # cv2.putText(frame,labels[class_id],(x+w+10,y+h),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
                    cv2.putText(frame,labels[class_id],(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
                    # Distance = sqrt((x+(w-x))**2 + (y+(h - y))**2)
                    # print("The distance between this two points is", str(round(Distance, 4))+" units")
                    if previousframe is not None:
                        if labels[class_id] == "SteelBall":
                            # print(" Ball Distance from the camera is ", str(round(x - prevx, 4))+" units")
                            if (x - prevx) > 0:
                                print("The ball is moving right")
                                print(" Ball Distance from the camera is ", str(round(x - prevx, 4))+" units")
                            elif (x - prevx) < 0:
                                print("The ball is moving left")
                                print(" Ball Distance from the camera is ", str(round(x - prevx, 4))+" units")
                            else:
                                print("The ball is not moving")
                            if (y - prevy) > 0:
                                print("The ball is moving down")
                            elif (y - prevy) < 0:
                                print("The ball is moving up")
                            else:
                                print("The ball is not moving")
                        previousframe = frame
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                        # Select good points
                        if p1 is not None:
                            good_new = p1[st==1]
                            good_old = p0[st==1]
                        # draw the tracks
                        for i, (new, old) in enumerate(zip(good_new, good_old)):
                            a, b = new.ravel()
                            c, d = old.ravel()
                            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
                        img = cv2.add(frame, mask)
                        imS = cv2.resize(img, (960, 540))                # Resize image
                        cv2.imshow("output", imS)
                        

    print(" Time taken = " + str(time.process_time() - start))
    
  
    # Display the resulting frame
    # cv2.imshow('frame', frame)
    # imS = cv2.resize(frame, (960, 540))                # Resize image
    # cv2.imshow("output", imS)
    previousframe = frame
    prevx = x
    prevy = y
    prevw = w
    prevh = h
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()


if __name__ == "__main__":
    main()