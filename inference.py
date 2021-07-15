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

import tools

def main():
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add("--model",help="Path to model")
    add("--checkpoint", type=tools.str2str_or_none, default=None,help="ckpt file holding saved state")
    add("--videopath",help="path to input file")
    add("--outputpath",help="name and path to output mp4 file")
    add("--step",type=int)
    args = parser.parse_args()
    
    #loading model and checkpoint
    model_dict=tools.module_classes_to_dict(models)
    if args.model not in model_dict:
        print("Model class not found")
        exit()
    else:
        model=model_dict[args.model](args=None)

    if args.checkpoint== None or not os.path.isfile(args.checkpoint):
        print("No checkpoint folder found")
        exit()
    else:
        checkpoint = torch.load(args.checkpoint)
        state_dict = checkpoint['state_dict']
        state_dict_new = {}
        for key, value in state_dict.items():
            key = key.replace("_model.", "")
            state_dict_new[key] = value
    #loading the state from the checkpoint onto the model
    model.load_state_dict(state_dict_new)
    model.cuda().eval()
    #
    if not os.path.isfile(args.videopath):
        print("Videopath is incorrect")
        exit()

    cap = cv2.VideoCapture(args.videopath)
    if not cap.isOpened():
        print("Unable to open video")
        exit()
    
    i=0 #indexing for temp images
    first_frame=True#Flag marking first frame for exclusion
    temp_path="temp" #path for temporary files to be stitched
    os.mkdir(temp_path)

    while(True):
        ret,frame=cap.read()
        if ret:
            if first_frame==True:
                first_frame=False
                prev_frame=frame
            else:
            #coverting frames to torch tensors and normalising
                im1_tensor = vision_transforms.transforms.ToTensor()(prev_frame).unsqueeze(0).cuda()
                im2_tensor = vision_transforms.transforms.ToTensor()(frame).unsqueeze(0).cuda()

                ## Forward pass
                input_dict = {'input1' : im1_tensor, 'input2' : im2_tensor}
                output_dict = model.forward(input_dict)
                flow = output_dict['flow'].squeeze(0).detach().cpu().numpy()
                flow_norm=np.linalg.norm(flow,axis=0)
                
                #Post processing flow
                #insert code for post processing here

                #plotting flow and saving figures to stitch into video
                step=args.step

                plt.figure()
                plt.axis('off')
                plt.imshow(prev_frame)
                plt.quiver(np.arange(0,flow.shape[2],step),np.arange(0,flow.shape[1],step),flow[0,::step, ::step],flow[1,::step,::step],flow_norm[::step,::step]+5)
                plt.savefig(os.path.join(temp_path,'combined_'+str(i)+'.png'),transparent=True,bbox_inches='tight', pad_inches=0)
                plt.close()
                i+=1

            prev_frame=frame# setting current frame to prev_frame for next iteration
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    # using ffmpeg to stitch photos in the temp folder into a
    os.system('ffmpeg -framerate 5 -i '+temp_path+'/combined_%d.png '+args.outputpath)
    shutil.rmtree(temp_path)

if __name__ == "__main__":
    main()