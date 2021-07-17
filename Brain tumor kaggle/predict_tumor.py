import torch
import os
import argparse
# from Tkinter import Tk     
from tkinter.filedialog import askopenfilename
import cv2
import torch
import numpy as np

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default='checkpoints/rs150-bs16', help='Path to checkpoint directory.')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use GPU.')
    return parser

def load(opts):
    model = torch.load(os.path.join(opts.load, 'model.pt'))
    return model

def predict(model, img, IMG_RESIZE):
    
    img = cv2.resize(img, (IMG_RESIZE,IMG_RESIZE))
    img = torch.from_numpy(np.array(img))
    img = img.float()
    output = model(img.view(-1,1,IMG_RESIZE,IMG_RESIZE))
    _, predictions = torch.max(output,1)
    return predictions

if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    model = load(opts)

    # Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename()
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    tumor = predict(model, img, 150)
    if tumor == 1:
        print('person has tumor')
    else:
        print('person do not have tumor')