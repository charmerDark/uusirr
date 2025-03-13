import models
import argparse
import os
import cv2
import tools
import shutil
import commandline
import torch
from torchvision import transforms as vision_transforms
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add("--model", help="Path to model", required=True)
    add("--checkpoint", type=tools.str2str_or_none, default=None, help="ckpt file holding saved state", required=True)
    add("--image1path", help="path to input file", required=True)
    add("--image2path", help="path to input file", required=True)
    add("--outputpath", help="name and path to output mp4 file, omit if output video not to be saved")
    add("--flowpath", help="add path to output .npy file, omit if flow not to be saved")
    add("--step", type=int)
    args = parser.parse_args()
    
    # Loading model and checkpoint
    model_dict = tools.module_classes_to_dict(models)
    if args.model not in model_dict:
        print("Model class not found")
        exit()
    else:
        model = model_dict[args.model](args=None)

    if args.checkpoint is None or not os.path.isfile(args.checkpoint):
        print("No checkpoint folder found")
        exit()
    else:
        checkpoint = torch.load(args.checkpoint)
        state_dict = checkpoint['state_dict']
        state_dict_new = {key.replace("_model.", ""): value for key, value in state_dict.items()}
    
    # Load state onto the model
    model.load_state_dict(state_dict_new)
    model.cuda().eval()

    # **Add code to read images and convert to tensor**
    if not os.path.isfile(args.image1path) or not os.path.isfile(args.image2path):
        print("Input image(s) not found")
        exit()

    # Read images using OpenCV
    prev_frame = cv2.imread(args.image1path)
    frame = cv2.imread(args.image2path)

    if prev_frame is None or frame is None:
        print("Failed to load input images")
        exit()

    # Convert BGR to RGB
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to tensor and normalize
    transform = vision_transforms.Compose([
        vision_transforms.ToTensor()
    ])
    
    im1_tensor = transform(prev_frame).unsqueeze(0).cuda()
    im2_tensor = transform(frame).unsqueeze(0).cuda()

    ## Forward pass
    input_dict = {'input1': im1_tensor, 'input2': im2_tensor}
    with torch.no_grad():
        output_dict = model.forward(input_dict)

    flow = output_dict['flow'].squeeze(0).detach().cpu().numpy()
    
    # Save flow if requested
    if args.flowpath:
        np.save(args.flowpath, flow)
        print(f"Flow saved to {args.flowpath}")

    # Plot and display flow
    plt.imshow(np.linalg.norm(flow, axis=0), cmap='viridis')
    plt.colorbar()
    plt.title('Optical Flow Magnitude')
    plt.show()

if __name__ == "__main__":
    main()
