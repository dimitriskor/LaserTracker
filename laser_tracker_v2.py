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


def gaussian(kernel_size = (9, 9), mu = 0, sigma = 1, domain = 4):
    if isinstance(kernel_size, int):
        k_x = kernel_size
        k_y = kernel_size 
    else:
        k_x = kernel_size[0]
        k_y = kernel_size[1]
    # circular gaussian (m_x, m_y) = (mu, mu)
    x = np.linspace(-domain, domain, k_x)
    y = np.linspace(-domain, domain, k_y)
    X, Y = np.meshgrid(x, y)
    return 1/ np.sqrt(np.pi) / sigma * np.exp(-0.5 * (((X - mu) ** 2) / (sigma ** 2) + ((Y - mu) ** 2) / (sigma ** 2)))



def gaussian_full(kernel_size = (640, 480), mu_x = 0, mu_y = 0, sigma = 1, domain_x = 640, domain_y = 480):
    if isinstance(kernel_size, int):
        k_x = kernel_size
        k_y = kernel_size 
    else:
        k_x = kernel_size[0]
        k_y = kernel_size[1]
    # circular gaussian (m_x, m_y) = (mu, mu)
    x = np.linspace(0, domain_x, k_x)
    y = np.linspace(0, domain_y, k_y)
    X, Y = np.meshgrid(x, y)
    gauss =  1/ np.sqrt(np.pi) / sigma * np.exp(-0.5 * (((X - mu_x) ** 2) / (sigma ** 2) + ((Y - mu_y) ** 2) / (sigma ** 2)))
    return gauss/np.max(gauss)# - 2*np.mean(gauss)

class LaserTracker(torch.nn.Module):
    def __init__(self, width = 640//4, height = 480//4):
        super(LaserTracker, self).__init__()
        self.pool = torch.nn.AvgPool2d(4)
        self.neural_state = torch.zeros((width, height))
        self.conv = torch.nn.Conv2d(1, 1, 9, stride = 1 , padding=4)
        DoG = gaussian(9, sigma=1) - gaussian(9, sigma=2)
        # plt.imshow(DoG)
        # plt.colorbar()
        # plt.savefig('sfg.png')
        # time.sleep(5)
        self.conv.weight.data = torch.FloatTensor(DoG).unsqueeze(0).unsqueeze(0)
        self.factor = 0.85
        self.prev_ind = None
    
    def forward(self, x):
        x = torch.tensor(x).unsqueeze(0).unsqueeze(0)
        # indices = divmod(self.neural_state.argmax().item(), self.neural_state.shape[1])
        # value = self.neural_state[indices[0], indices[1]] * 0.1
        x = self.pool(x)
        indices = divmod(x.argmax().item(), x.shape[1])
        # try:
            # if indices_x[0] - self.prev_ind[0] < 60 and indices_x[1] - self.prev_ind[1] < 60:
                # x = 3*x
        # except:
            # pass
        # gauss = gaussian_full(mu_x=indices[0], mu_y=indices[1], sigma=4)
        if self.prev_ind != None:
            gauss_prev = torch.tensor(gaussian_full(kernel_size = (640//4, 480//4),mu_x=self.prev_ind[0], mu_y=self.prev_ind[1], sigma=1000).T).unsqueeze(0).unsqueeze(0)
        #     # gauss_prev[self.prev_ind[0], self.prev_ind[1]] = 0
            self.prev_ind = indices
        else:
            self.prev_ind = indices
            gauss_prev = torch.zeros_like(self.neural_state)
        # gauss = gaussian_full(mu_x=indices[0], mu_y=indices[1], sigma=1*(480-indices[1])/480)
        # plt.imshow(gauss)
        # plt.colorbar()
        # plt.savefig('sf.png')
        # plt.close()
        # print(self.neural_state.shape)
        # print(gauss_prev.shape)
        # print(x.shape)
        print(torch.sum(gauss_prev))
        self.neural_state = self.neural_state*self.factor + (0.2+gauss_prev)*x
        # self.neural_state = self.neural_state*self.factor*(gauss_prev.T+0.1) + x*(gauss.T+0.1)#+ self.conv(x)[0][0] -value
        # self.neural_state = self.conv(x.unsqueeze(0).unsqueeze(0))
        return self.neural_state



if __name__ == "__main__":

    tracker = LaserTracker()
    tracker.eval()
    stat = []

    win_name = "Durin"
    cv2.namedWindow(win_name)        # Create a named window
    # cv2.moveWindow(win_name, 640, 200)  # Move it to (40,30)
    scale = 2


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
            with Durin('durin0.local') as durin:
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
                            # if np.sum(frame) > 4*mean_val or np.sum(frame) == 0.0:
                            #     continue
                        # print(mean_val)
                        frame_inter =  tracker(frame)
                        frame_viz[:,:,1] = frame_inter.numpy()
                        # frame_viz = frame_viz.numpy()
                        indices = divmod(frame_viz[:,:,1].argmax().item(), frame_viz.shape[1])
                        # frame = tracker(frame)
                        prev_events = np.sum(frame)
                        # print(np.sum(frame_viz))
                        new_width = math.ceil(640 * scale)
                        new_height = math.ceil(480 * scale)

                        # Ensure the target size is valid
                        if new_width <= 0 or new_height <= 0:
                            raise ValueError(f"Invalid target size: width={new_width}, height={new_height}. Ensure that the scale factor is valid.")
                        image = cv2.resize(frame_viz.transpose(1, 0, 2), (new_width, new_height), interpolation = cv2.INTER_AREA)
                        cv2.imshow(win_name, image)
                        cv2.waitKey(1)
                        (obs, dvs, cmd) = durin.read()
                        min_tof = False
                        print(obs.charge)
                        # eds
                        for t in obs.tof:
                            if np.mean(t) < 300 and np.mean(t) != 0:
                                min_tof = True
                        if min_tof:
                            durin(Move(0,0,0))
                            break
                        # print(frame[indices])
                        if frame_viz[indices[0], indices[1], 1] > 2:
                            if indices[1] < 180//4:
                                continue
                            ind_x = indices[1] * 2 - 410//4
                            complete_motion_r.append(np.clip(-1*(indices[0]-320//4), -300, 300))
                            complete_motion_x.append(np.clip(3.5*4*(ind_x - 410//4), -300, 300))
                            counter_xy += 1
                            if counter_xy >= 15:
                                complete_motion_r.pop(0)
                                complete_motion_x.pop(0)
                                counter_xy -= 1
                            # print(sum(complete_motion_x)/15, sum(complete_motion_r)/15)
                            print("YES")
                            durin(Move(0, sum(complete_motion_x)/15, sum(complete_motion_r)/15))             # LaserTracker on the ground
                            # durin(Move(0,0,0))
                        else:
                            if counter_xy > 0:
                                complete_motion_r.pop(0)
                                complete_motion_x.pop(0)
                                counter_xy -= 1
                                durin(Move(0, sum(complete_motion_x)/15, sum(complete_motion_r)/15))             # LaserTracker on the ground
                            else:
                                durin(Move(0,0,0))
                            print("N0")

                            # durin(Move(sum(complete_motion_x)/5, sum(complete_motion_y)/5, 0))
                        # print('here')
                        lag = time.time()-st
                        # print(lag)
                        # # if lag < 0.005:
                        #     time.sleep(0.005-lag)
                        print(lag)
                        # print('Mean loop delay:', torch.mean(torch.tensor(stat)))
                except Exception as e:
                    print(e)
