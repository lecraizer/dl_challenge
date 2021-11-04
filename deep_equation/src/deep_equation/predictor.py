"""
Predictor interfaces for the Deep Learning challenge.
"""
import json
import torch
from typing import List
import numpy as np
from torchvision import transforms
from deep_equation.model import CNN


class BaseNet:
    """
    Base class that must be used as base interface to implement 
    the predictor using the model trained by the student.
    """

    def load_model(self, model_path):
        """
        Implement a method to load models given a model path.
        """
        pass

    def predict(
        self, 
        images_a: List, 
        images_b: List, 
        operators: List[str], 
        device: str = 'cpu'
    ) -> List[float]:
        """
        Make a batch prediction considering a mathematical operator 
        using digits from image_a and image_b.
        Instances from iamges_a, images_b, and operators are aligned:
            - images_a[0], images_b[0], operators[0] -> regards the 0-th input instance
        Args: 
            * images_a (List[PIL.Image]): List of RGB PIL Image of any size
            * images_b (List[PIL.Image]): List of RGB PIL Image of any size
            * operators (List[str]): List of mathematical operators from ['+', '-', '*', '/']
                - invalid options must return `None`
            * device: 'cpu' or 'cuda'
        Return: 
            * predicted_number (List[float]): the list of numbers representing the result of the equation from the inputs: 
                [{digit from image_a} {operator} {digit from image_b}]
        """
    # do your magic

    pass 


class RandomModel(BaseNet):
    """This is a dummy random classifier, it is not using the inputs
        it is just an example of the expected inputs and outputs
    """

    def load_model(self, model_path):
        """
        Method responsible for loading the model.
        If you need to download the model, 
        you can download and load it inside this method.
        """
        np.random.seed(42)

    def predict(
        self, images_a, images_b,
        operators, device = 'cpu'
    ) -> List[float]:

        predictions = []
        for image_a, image_b, operator in zip(images_a, images_b, operators):            
            random_prediction = np.random.uniform(-10, 100, size=1)[0]
            predictions.append(random_prediction)
        
        return predictions

    
class StudentModel(BaseNet):
    """
    TODO: THIS is the class you have to implement:
        load_model: method that loads your best model.
        predict: method that makes batch predictions.
    """

    # TODO
    def load_model(self, model_path='deep_equation/input/model_20epochs.pth'):
        """
        Load the student's trained model.
        TODO: update the default `model_path` 
              to be the correct path for your best model!
        """

        cnn = CNN()
        cnn.load_state_dict(torch.load(model_path))

        return cnn
    
    # TODO:
    def predict(
        self, images_a, images_b,
        operators, device = 'cpu'
    ):
        """Implement this method to perform predictions 
        given a list of images_a, images_b and operators.
        """
        with open('deep_equation/src/deep_equation/labels_dict.json', 'r') as fp:
            mapping_dict = json.loads(fp.read())
            
        with open('deep_equation/src/deep_equation/inv_labels_dict.json', 'r') as fp:
            inv_labels_dict = json.loads(fp.read())
        
        symbols = ['+', '-', '*', '/']
        op_dict = {'+': [1, 0, 0, 0], '-': [0, 1, 0, 0], '*': [0, 0, 1, 0], '/': [0, 0, 0, 1]}
        
        cnn = self.load_model()
        
        predictions = []
        for i in range(len(images_a)):
            img1 = images_a[i]
            img2 = images_b[i]
            op = operators[i]
            img_to_tensor = transforms.Compose([
                transforms.Resize((32, 32)), 
                transforms.Grayscale(), 
                transforms.ToTensor(),
            ])

            a = img_to_tensor(img1)
            b = img_to_tensor(img2)

            T = torch.stack((a, b))

            v = op_dict[op]
            V = torch.tensor([1*[32*[8*v]]])

            T = torch.cat((T, V), 0) 
            T = T.permute(1, 0, 2, 3)

            cnn.eval()
            with torch.no_grad():
                test_output, last_layer = cnn(T)            
                pred_y = torch.max(test_output, 1)[1].data.squeeze().item()
                result = inv_labels_dict[str(pred_y)]

            predictions.append(float(result))
        
        print(predictions)
        return predictions
