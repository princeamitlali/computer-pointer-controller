\
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork,IECore
import warnings
warnings.filterwarnings("ignore")

class FaceDetectionClass:

    def __init__(self, model_name, device, threshold, extensions=None):
 
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + ".xml"
        self.device = device
        self.threshold = threshold
        self.extension = extensions
        self.cropped_face_image = None
        self.first_face_coordinates = None
        self.face_coords = None
        self.results = None
        self.pre_image = None
        self.net = None

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
        self.results = self.net.infer({self.input_name: self.pre_image})
        self.face_coords = self.preprocess_output(self.results, image)

        if len(self.face_coords) == 0:
            log.error("No Face is detected, Next frame will be processed..")
            return 0, 0

        self.first_face_coordinates = self.face_coords[0]
        cropped_face_image = image[self.first_face_coordinates[1]:self.first_face_coordinates[3],
                             self.first_face_coordinates[0]:self.first_face_coordinates[2]]

        return self.first_face_coordinates, cropped_face_image

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):

        pre_frms = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        pre_frms = pre_frms.transpose((2, 0, 1))
        pre_frms = pre_frms.reshape(1, *pre_frms.shape)

        return pre_frms

    def preprocess_output(self, outputs,image):

        face_coords = []
        outs = outputs[self.output_name][0][0]
        for box in outs:
            conf = box[2]
            if conf >= self.threshold:
                x_min = int(box[3] * image.shape[1])
                y_min = int(box[4] * image.shape[0])
                x_max = int(box[5] * image.shape[1])
                y_max = int(box[6] * image.shape[0])
                face_coords.append([x_min, y_min, x_max, y_max])
        return face_coords
