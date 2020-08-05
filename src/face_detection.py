from openvino.inference_engine import IECore
import numpy as np
import cv2


class FaceDetection:
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, threshold=0.8, device='CPU', precision='FP32'):
        self.threshold = threshold
        self.device = device
        self.exec_net = None

        try:
            path = f"models/intel/{model_name}/{precision}"
            model_weights = f"{path}/{model_name}.bin"
            model_structure = f"{path}/{model_name}.xml"
            self.core = IECore()
            self.net = self.core.read_network(
                model=model_structure, weights=model_weights)
        except:
            raise ValueError("Could not initialize the network")

        self.input_name = next(iter(self.net.inputs))
        self.input_shape = self.net.inputs[self.input_name].shape
        self.output_name = next(iter(self.net.outputs))
        self.output_shape = self.net.outputs[self.output_name].shape

    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.exec_net = self.core.load_network(
            network=self.net, device_name=self.device, num_requests=1)

    def predict(self, image):
        image = self.preprocess_input(image)
        result = self.exec_net.infer({self.input_name: image})
        coords = self.preprocess_output(result)

        return coords

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Net input shape = [1x3x384x672]
        An input image in the format [BxCxHxW]
        '''
        *_, height, width = self.input_shape
        image = cv2.resize(image, (width, height))
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, :, :, :]

        return image

    def preprocess_output(self, outputs):
        coords = outputs[self.output_name]
        for coord in coords[0][0]:
            if coord[2] >= self.threshold:
                x_min = coord[3]
                y_min = coord[4]
                x_max = coord[5]
                y_max = coord[6]

        return [x_min, y_min, x_max, y_max]
