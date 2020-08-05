
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IECore
import warnings
import math
warnings.filterwarnings("ignore")

class GazeEstimationClass:
    
    def __init__(self, model_name, device='CPU', extensions=None):
        
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extension = extensions

        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name = next(iter(self.model.inputs))
        # self.input_shape = self.model.inputs[self.input_name['left_eye_image']].shape
        self.output_name = next(iter(self.model.out))
        self.output_shape = self.model.out[self.output_name].shape

    def load_model(self):
       
        self.model = IENetwork(self.model_structure, self.model_weights)
        self.core = IECore()
        sup_layer = self.core.query_network(network=self.model, device_name=self.device)
        unsup_layer = [R for R in self.model.layers.keys() if R not in sup_layer]

        if len(unsup_layer) != 0:
            log.error("Unsupported layers found ...")
            log.error("Adding specified extension")
            self.core.add_extension(self.extension, self.device)
            sup_layer = self.core.query_network(network=self.model, device_name=self.device)
            unsup_layer = [R for R in self.model.layers.keys() if R not in sup_layer]
            if len(unsup_layer) != 0:
                log.error("ERROR: There are still unsupported layers after adding extension...")
                exit(1)
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)

    def predict(self, left_eye_image, right_eye_image, head_pose_output):
        
        self.l_eye_pre_img, self.r_eye_pre_img = self.preprocess_input(left_eye_image, right_eye_image)
        self.results = self.net.infer(
            inputs={'left_eye_image': self.l_eye_pre_img, 'right_eye_image': self.r_eye_pre_img,
                    'head_pose_angles': head_pose_output})
        self.mouse_coordinate, self.gaze_vector = self.preprocess_output(self.results, head_pose_output)

        return self.mouse_coordinate, self.gaze_vector

    def check_model(self):
        pass

    def preprocess_input(self, left_eye_image, right_eye_image):
        
        l_eye_pre_img = cv2.resize(left_eye_image, (60, 60))
        l_eye_pre_img = l_eye_pre_img.transpose((2, 0, 1))
        l_eye_pre_img = l_eye_pre_img.reshape(1, *l_eye_pre_img.shape)

        r_eye_pre_img = cv2.resize(right_eye_image, (60, 60))
        r_eye_pre_img = r_eye_pre_img.transpose((2, 0, 1))
        r_eye_pre_img = r_eye_pre_img.reshape(1, *r_eye_pre_img.shape)

        return l_eye_pre_img, r_eye_pre_img

    def preprocess_output(self, out, head_pose_estimation_output):
       
        roll_val = head_pose_estimation_output[2]
        out = out[self.output_name][0]
        cos_theta = math.cos(roll_val * math.pi / 180)
        sin_theta = math.sin(roll_val * math.pi / 180)

        x_val = out[0] * cos_theta + out[1] * sin_theta
        y_val = out[1] * cos_theta - out[0] * sin_theta

        return (x_val, y_val), out
