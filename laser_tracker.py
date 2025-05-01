import aestream
from durin import *
import numpy as np 
import torch
import argparse
import time
import cv2
import math
import matplotlib.pyplot as plt

args = argparse.ArgumentParser()
args.add_argument('-d', '--durin', default='durin0.local', help='Durin IP')
args.add_argument('-p', '--port', default='4001', help='Port to fetch events')
args = args.parse_args()


class LaserTracker(torch.nn.Module):
    def __init__(self, width = 640//4, height = 480//4):
        super(LaserTracker, self).__init__()
        self.pool = torch.nn.AvgPool2d(4)
        self.neural_state = torch.zeros((width, height))
        self.factor = 0.94
        self.prev_ind = None
    
    def forward(self, x):
        x = torch.tensor(x).unsqueeze(0).unsqueeze(0)
        indices = divmod(self.neural_state.argmax().item(), self.neural_state.shape[1])
        # value = self.neural_state[indices[0], indices[1]] * 0.1
        x = self.pool(x)
        indices_x = divmod(x.argmax().item(), x.shape[1])
        try:
            if indices_x[0] - self.prev_ind[0] < 60 and indices_x[1] - self.prev_ind[1] < 60:
                x = 3*x
        except:
            pass
        self.neural_state = self.neural_state*self.factor + x
        return self.neural_state



if __name__ == "__main__":

    tracker = LaserTracker()
    tracker.eval()
    stat = []

    win_name = "Durin"
    cv2.namedWindow(win_name)        # Create a named window
    # cv2.moveWindow(win_name, 640, 200)  # Move it to (40,30)
    scale = 1


    # Stream events from UDP port 3333 (default)
    frame_viz = np.zeros((640//4,480//4, 3))
    counter = 0.0 
    mean_val = 0.0 
    mean_list = []
    counter_xy = 0
    counter_noise_calc = 0
    complete_motion_x = []
    complete_motion_r = []
    with torch.no_grad():
        with aestream.UDPInput((640, 480), device = 'cpu', port=args.port) as stream:
            with Durin('durin1.local') as durin:
                try:
                    while True:
                        counter_noise_calc += 1
                        st = time.time()
                        # frame = (stream.read('torch'))
                        frame = (stream.read())

                        counter += 1
                        mean_list.append(np.sum(frame))
                        mean_val += np.sum(frame)/100
                        if counter >= 100:
                            mean_val -= mean_list[0]/100
                            mean_list.pop(0)
                            if np.sum(frame) > 4*mean_val or np.sum(frame) == 0.0:
                                continue
                        frame_inter =  tracker(frame)
                        frame_viz[:,:,1] = frame_inter.numpy()
                        indices = divmod(frame_viz[:,:,1].argmax().item(), frame_viz.shape[1])
                        prev_events = np.sum(frame)
                        new_width = math.ceil(640 * scale)
                        new_height = math.ceil(480 * scale)

                        # Ensure the target size is valid
                        if new_width <= 0 or new_height <= 0:
                            raise ValueError(f"Invalid target size: width={new_width}, height={new_height}. Ensure that the scale factor is valid.")
                        image = cv2.resize(frame_viz.transpose(1, 0, 2)/np.max(frame_viz+1), (new_width, new_height), interpolation = cv2.INTER_AREA)
                        cv2.imshow(win_name, image)
                        cv2.waitKey(1)
                        (obs, dvs, cmd) = durin.read()
                        min_tof = False
                        print(obs.charge)
                        for t in obs.tof:
                            if np.mean(t) < 300 and np.mean(t) != 0:
                                min_tof = True
                        if min_tof:
                            durin(Move(0,0,0))
                            break
                        if frame_viz[indices[0], indices[1], 1] > 1.2:
                            if indices[1] < 180//4:
                                continue
                            complete_motion_r.append(np.clip(-1*(indices[0]-320//4), -200, 200))
                            complete_motion_x.append(np.clip(3.5*4*(indices[1]-410//4), -200, 200))
                            counter_xy += 1
                            if counter_xy >= 15:
                                complete_motion_r.pop(0)
                                complete_motion_x.pop(0)
                                counter_xy -= 1
                            print("YES")
                            durin(Move(0, sum(complete_motion_x)/15, sum(complete_motion_r)/15))             # LaserTracker on the ground
                        else:
                            if counter_xy > 0:
                                complete_motion_r.pop(0)
                                complete_motion_x.pop(0)
                                counter_xy -= 1
                                durin(Move(0, sum(complete_motion_x)/15, sum(complete_motion_r)/15))             # LaserTracker on the ground
                            else:
                                durin(Move(0,0,0))
                            print("N0")

                        # print(lag)
                except Exception as e:
                    print(e)
