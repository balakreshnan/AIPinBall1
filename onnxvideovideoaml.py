import argparse
import pathlib
import numpy as np
import onnxruntime
import onnxruntime as ort
import PIL.Image
import time
import pytesseract
from PIL import Image
import cv2
import onnx
from math import sqrt
import math
import json
from yolo_onnx_preprocessing_utils import preprocess, preprocess1, frame_resize
from yolo_onnx_preprocessing_utils import non_max_suppression, _convert_to_rcnn_output
import torch
from datetime import datetime
import os
#from objdict import ObjDict
from yolo_onnx_preprocessing_utils import letterbox, non_max_suppression, _convert_to_rcnn_output
from onnxruntime_object_detection import ObjectDetection
import tempfile
#import tensorflow as tf
from torchvision import transforms
from json_tricks import dumps


PROB_THRESHOLD = 0.40  # Minimum probably to show results.

print(" Onnx Runtime : " + onnxruntime.get_device())

#labels = ['ballfail','ballinendzone','flaphit','SteelBall','zone1','zone2']
# labels = ['ballfail','ballinendzone','ballinplunge','flaphit','SteelBall','zone1','zone2']
#labels = ['BallinPlunger','EndZone','FailZone','GameOver','leftflap','RightFlap','SteelBall']

labels_file = "steelball1aml/labels.json"
onnx_model_path = "steelball1aml/model.onnx"

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

with open(labels_file) as f:
    labels = json.load(f)
print(labels)


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

    def predict1(self, image_filepath):
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
        return outputs[0]


    def print_outputs(outputs):
        assert set(outputs.keys()) == set(['detected_boxes', 'detected_classes', 'detected_scores'])
        for box, class_id, score in zip(outputs['detected_boxes'][0], outputs['detected_classes'][0], outputs['detected_scores'][0]):
            if score > PROB_THRESHOLD:
                # print("{class_id}")
                # print(f"Az Cog Label: {class_id}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")
            # print(f"Az Cog Label: {labels[class_id]}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")
                if (class_id >= 0 and class_id <= 3):
                    print(f"Az Cog Label: {labels[class_id]}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")

def get_predictions_from_ONNX(onnx_session,img_data):
    """perform predictions with ONNX Runtime
    
    :param onnx_session: onnx model session
    :type onnx_session: class InferenceSession
    :param img_data: pre-processed numpy image
    :type img_data: ndarray with shape 1xCxHxW
    :return: boxes, labels , scores 
    :rtype: list
    """
    sess_input = onnx_session.get_inputs()
    sess_output = onnx_session.get_outputs()
    # predict with ONNX Runtime
    output_names = [ output.name for output in sess_output]
    # print(output_names)
    pred = onnx_session.run(output_names=output_names, input_feed={sess_input[0].name: img_data})
    return pred[0]

def _get_box_dims(image_shape, box):
    box_keys = ['topX', 'topY', 'bottomX', 'bottomY']
    height, width = image_shape[0], image_shape[1]

    box_dims = dict(zip(box_keys, [coordinate.item() for coordinate in box]))

    box_dims['topX'] = box_dims['topX'] * 1.0 / width
    box_dims['bottomX'] = box_dims['bottomX'] * 1.0 / width
    box_dims['topY'] = box_dims['topY'] * 1.0 / height
    box_dims['bottomY'] = box_dims['bottomY'] * 1.0 / height

    return box_dims

def _get_prediction(label, image_shape, classes):
    
    boxes = np.array(label["boxes"])
    labels = np.array(label["labels"])
    labels = [label[0] for label in labels]
    scores = np.array(label["scores"])
    scores = [score[0] for score in scores]

    bounding_boxes = []
    for box, label_index, score in zip(boxes, labels, scores):
        box_dims = _get_box_dims(image_shape, box)

        box_record = {'box': box_dims,
                      'label': classes[label_index],
                      'score': score.item()}

        bounding_boxes.append(box_record)

    return bounding_boxes


model_path = "steelball1aml/model.onnx"

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
movingleft = 0
movingright = 0
movingup = 0
movingdown = 0
distance = 0

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

session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)

sess_input = session.get_inputs()
sess_output = session.get_outputs()
print(f"No. of inputs : {len(sess_input)}, No. of outputs : {len(sess_output)}")

for idx, input_ in enumerate(range(len(sess_input))):
    input_name = sess_input[input_].name
    input_shape = sess_input[input_].shape
    input_type = sess_input[input_].type
    print(f"{idx} Input name : { input_name }, Input shape : {input_shape}, \
    Input type  : {input_type}")  

for idx, output in enumerate(range(len(sess_output))):
    output_name = sess_output[output].name
    output_shape = sess_output[output].shape
    output_type = sess_output[output].type
    print(f" {idx} Output name : {output_name}, Output shape : {output_shape}, \
    Output type  : {output_type}")

batch, channel, height_onnx, width_onnx = session.get_inputs()[0].shape
#batch, channel, height_onnx, width_onnx

print(session.get_inputs()[0].shape)

from onnxruntime_yolov5 import initialize_yolov5
labelPath = f'steelball1aml/labels.json'
labelFile = 'labels.json'
initialize_yolov5(onnx_model_path, labelPath, 640,0.4,0.5) 

frame_size = (960, 540)
# Initialize video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# output = cv2.VideoWriter('C:\\Users\\babal\\Downloads\\output_video_from_file.mp4', fourcc, 60, frame_size, 1)
output = cv2.VideoWriter('C:\\Users\\babal\\Downloads\\output1.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20, frame_size)

batch_size = session.get_inputs()[0].shape
print(labels)

# Read until video is completed 
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    start = time.process_time()
    #outputs = model.predict1(frame)
    #print(outputs)
    #outputs = model.predict1(frame)
    #image = PIL.Image.fromarray(frame, 'RGB').resize(640,640)
    #assert batch_size == frame.shape[0]
    #print(session.get_inputs()[0].shape[2:])
    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # image = Image.fromarray(img)
    #image = PIL.Image.fromarray(img, 'RGB').resize(frame.input_shape)
    #input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
    preprocessimg = preprocess1(frame)
    #convert_tensor = transforms.ToTensor()

    img = cv2.resize(frame, (640, 640))
    # convert image to numpy
    # print(session.get_inputs()[0].shape)
    x = np.array(img).astype('float32').reshape([1, channel, height_onnx, width_onnx])
    x = x / 255

    # y = preprocessimg.tolist()
    # result = get_predictions_from_ONNX(session, preprocessimg)

    #preprocessimg = frame_resize(frame)

    #result = get_predictions_from_ONNX(session, x)
    #print(result)
    h, w = frame.shape[:2]

    frame_optimized, ratio, pad_list = frame_resize(frame, 640)
    from onnxruntime_yolov5 import predict_yolov5
    result = predict_yolov5(frame_optimized, pad_list)
    predictions = result['predictions'][0]
    new_w = int(ratio[0]*w)
    new_h = int(ratio[1]*h)
    frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    annotated_frame = frame_resized.copy()

    #print(predictions)

    detection_count = len(predictions)
    #print(f"Detection Count: {detection_count}")

    if detection_count > 0:
        for i in range(detection_count):
            bounding_box = predictions[i]['bbox']
            tag_name = predictions[i]['labelName']
            probability = round(predictions[i]['probability'],2)
            image_text = f"{probability}%"
            color = (0, 255, 0)
            thickness = 1
            xmin = int(bounding_box["left"])
            xmax = int(bounding_box["width"])
            ymin = int(bounding_box["top"])
            ymax = int(bounding_box["height"])
            start_point = (int(bounding_box["left"]), int(bounding_box["top"]))
            end_point = (int(bounding_box["width"]), int(bounding_box["height"]))
            annotated_frame = cv2.rectangle(annotated_frame, start_point, end_point, color, thickness)
            cv2.putText(annotated_frame,tag_name + '-' + image_text,(xmin-10,ymin-10),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
            imS = cv2.resize(annotated_frame, (960, 540))
            cv2.imshow('frame', imS)


    print(" Time taken = " + str(time.process_time() - start))

    # imS = cv2.resize(img, (960, 540))

    # Display the resulting frame
    # cv2.imshow('frame', imS)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
output.release()
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()


if __name__ == "__main__":
    main()