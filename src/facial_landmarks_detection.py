
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IECore
import warnings
warnings.filterwarnings("ignore")

class FacialLandmarksClass:


    def __init__(self, model_name, device, extensions=None):

        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extension = extensions

        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

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

    def predict(self, image):

        self.pre_image = self.preprocess_input(image)
        self.results = self.net.infer(inputs={self.input_name: self.pre_image})
        self.output = self.preprocess_output(self.results, image)
        l_eye_x_min = self.output['l_eye_x_coord'] - 10
        l_eye_x_max = self.output['l_eye_x_coord'] + 10
        l_eye_y_min = self.output['l_eye_y_coord'] - 10
        l_eye_y_max = self.output['l_eye_y_coord'] + 10

        r_eye_x_min = self.output['r_eye_x_coord'] - 10
        r_eye_x_max = self.output['r_eye_x_coord'] + 10
        r_eye_y_min = self.output['r_eye_y_coord'] - 10
        r_eye_y_max = self.output['r_eye_y_coord'] + 10

        self.eye_coord = [[l_eye_x_min, l_eye_y_min, l_eye_x_max, l_eye_y_max],
                          [r_eye_x_min, r_eye_y_min, r_eye_x_max, r_eye_y_max]]
        l_eye_img = image[l_eye_x_min:l_eye_x_max, l_eye_y_min:l_eye_y_max]
        r_eye_img = image[r_eye_x_min:r_eye_x_max, r_eye_y_min:r_eye_y_max]

        return l_eye_img, r_eye_img, self.eye_coord

    def check_model(self):
        pass

    def preprocess_input(self, image):

        pre_frms = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        pre_frms = pre_frms.transpose((2, 0, 1))
        pre_frms = pre_frms.reshape(1, *pre_frms.shape)

        return pre_frms

    def preprocess_output(self, outputs, image):

        outputs = outputs[self.output_name][0]
        l_eye_x_coord = int(outputs[0] * image.shape[1])
        l_eye_y_coord = int(outputs[1] * image.shape[0])
        r_eye_x_coord = int(outputs[2] * image.shape[1])
        r_eye_y_coord = int(outputs[3] * image.shape[0])

        return {'left_eye_x_coordinates': l_eye_x_coord, 'left_eye_y_coordinates': l_eye_y_coord,
                'rright_eye_x_coordinates': r_eye_x_coord, 'right_eye_y_coordinates': r_eye_y_coord}
