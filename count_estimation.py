import torch
import utils as ut
import torchvision.transforms.functional as FT
from skimage.io import imread,imsave
from torchvision import transforms
from models import model_dict
from skimage import data,segmentation,measure,morphology,color,draw,data
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from timer import Timer
import numpy as np
import os


model_name = 'ResUnet'
model_path = './oilpalm_checkpoints/ResUnet/best_model_oilpalm_ResUnet.pth'
test_path = '/mnt/a409/users/tongpinmo/projects/crowdcount-mcnn/data-oilpalm/oilpalm-test/'
gt_file = os.path.join(test_path,'gt_palm.txt')
transformer = ut.ComposeJoint(
                [
                     [transforms.ToTensor(), None],
                     [transforms.Normalize(*ut.mean_std), None],
                     [None,  ut.ToLong() ]
                ])

# Load best model
model = model_dict[model_name](n_classes=2).cuda()
model.load_state_dict(torch.load(model_path))

mae = 0.0
rmse = 0.0
mrmse = 0.0

with open(gt_file,'r') as f:
    lines = f.readlines()
    samples = len(lines)
    print('samples: ',samples)
    for line in lines:
        line = line.strip('').split(' ')
        image_name = os.path.basename(line[0])
        image_path = os.path.join(test_path,'IMG_Palm',image_name)
        # print('image_path: ',image_path)

        image_raw = imread(image_path)
        collection = list(map(FT.to_pil_image, [image_raw, image_raw]))
        image, _ = transformer(collection)

        batch = {"images":image[None]}

        gt_count = int(line[1])
        print('gt_count: ',gt_count)
        et_count = int(model.predict(batch, method="counts").ravel()[0])
        print('pred_counts: ',et_count)

        mae += abs(gt_count - et_count)
        rmse += ((gt_count - et_count) * (gt_count - et_count))

    mae = mae / samples
    rmse = np.sqrt(rmse / samples)
    print('rmse: ', rmse)
    mrmse = np.sqrt(rmse / samples).mean()
    print('\nMAE: %0.3f, RMSE: %0.3f' % (mae, rmse))
    print('\nmRMSE:%0.3f' % (mrmse))










