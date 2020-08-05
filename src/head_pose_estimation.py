
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IECore
import warnings
warnings.filterwarnings("ignore")

class HeadPoseEstimationClass:

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
        self.output_list = self.preprocess_output(self.results)
        return self.output_list

    def check_model(self):
        pass

    def preprocess_input(self, image):
       
        pre_frms = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        pre_frms = pre_frms.transpose((2, 0, 1))
        pre_frms = pre_frms.reshape(1, *pre_frms.shape)
        return pre_frms

    def preprocess_output(self, outputs):
       
        out = []
        out.append(outputs['angle_y_fc'].tolist()[0][0])
        out.append(outputs['angle_p_fc'].tolist()[0][0])
        out.append(outputs['angle_r_fc'].tolist()[0][0])
        return out
