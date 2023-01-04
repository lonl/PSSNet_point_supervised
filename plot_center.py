import cv2
import os

filepath = '/mnt/a409/users/tongpinmo/projects/LCFCN/datasets/acacia-train/IMG_point/IMG_0001.txt'
imgpath = '/mnt/a409/users/tongpinmo/projects/LCFCN/figures/IMG_0001.jpg'

save_path = os.path.basename(imgpath)
save_path = os.path.join('/mnt/a409/users/tongpinmo/projects/LCFCN/figures',save_path.split('.')[0]+'_gt.jpg')
print('save_path: ',save_path)

with open(filepath,'r') as tf:
    lines = tf.readlines()
    img = cv2.imread(imgpath)

    for line in lines:
        line = line.strip().split(' ')
        print('line:',line)
        print(line[0],line[1])
        x = int(line[0])
        y = int(line[1])

        point = (x,y)
        radius = 2
        thickness = 6
        color = (0,255,255)  #BGR

        cv2.circle(img,point,radius,color,thickness)
    cv2.imwrite(save_path,img)
    



