import aestream
from durin import *
import numpy as np 
import torch
import argparse
import time
import cv2
import math

args = argparse.ArgumentParser()
args.add_argument('-d', '--durin', default='durin0.local', help='Durin IP')
args.add_argument('-p', '--port', default='4001', help='Port to fetch events')
args = args.parse_args()

class LaserTracker(torch.nn.Module):
    def __init__(self, width = 640, height = 480):
        super(LaserTracker, self).__init__()
        self.neural_state = torch.zeros((width, height))
        self.factor = 0.6
    
    def forward(self, x):
        self.neural_state = self.neural_state*self.factor + x
        return self.neural_state


tracker = LaserTracker()
stat = []

# win_name = "Durin"
# cv2.namedWindow(win_name)        # Create a named window
# cv2.moveWindow(win_name, 640, 200)  # Move it to (40,30)
# scale = 2


# Stream events from UDP port 3333 (default)
frame_viz = np.zeros((640,480*1,3))

if __name__ == "__main__":

    with aestream.UDPInput((640, 480), device = 'cpu', port=args.port) as stream:
        with Durin('durin0.local') as durin:
            try:
                while True:
                    st = time.time()
                    frame = tracker(stream.read())
                    indices = divmod(frame.argmax().item(), frame.shape[1])
                    # frame_viz[:,:,1] = frame
                    # image = cv2.resize(frame_viz.transpose(1,0,2), (math.ceil(640*2*scale),math.ceil(480*1*scale)), interpolation = cv2.INTER_AREA)
                    # cv2.imshow(win_name, image)
                    if frame[indices] > 6:
                        # durin(Move(-0.8*(indices[0]-320), 0.8*(indices[1]-380), 0))             # LaserTracker on the ground
                        durin(Move(-(indices[0]-320), 0, 0))                                      # LaserTracker in wall
                    else:
                        durin(Move(0,0,0))
                    print(indices)
                    # cv2.waitKey(1)
                    lag = time.time()-st
                    stat.append(lag)

                    # print('Mean loop delay:', torch.mean(torch.tensor(stat)))
            except Exception as e:
                print(e)

