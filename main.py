

import cv2
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np


def generate_dataset(type):

    if type == "test":
        return [
            {
                'path_img':'data/test/aa.png'
            },
           # {
           #     'path_img':'data/test/finger_print.png'
           # },
            {
                'path_img':'data/test/moon.jpeg'
            }
            
        ]
    elif type == "seeds":
        return [
                { 
                    'path_img': 'data/dataset_seeds/imagens/0100.jpg', 
                    'path_xml':'data/dataset_seeds/gabarito/0100.xml' 
                },
                { 
                    'path_img': 'data/dataset_seeds/imagens/0030.jpg', 
                    'path_xml':'data/dataset_seeds/gabarito/0030.xml' 
                },
                { 
                    'path_img': 'data/dataset_seeds/imagens/0020.jpg', 
                    'path_xml':'data/dataset_seeds/gabarito/0020.xml' 
                },
                { 
                    'path_img': 'data/dataset_seeds/imagens/0010.jpg', 
                    'path_xml':'data/dataset_seeds/gabarito/0010.xml' 
                },
                { 
                    'path_img': 'data/dataset_seeds/imagens/0001.jpg', 
                    'path_xml':'data/dataset_seeds/gabarito/0001.xml' 
                }
        ]
    else:
        reuturn = []


def main():
    print("")

if __name__ == "__main__":
    main()