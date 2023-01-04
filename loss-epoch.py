import matplotlib
import matplotlib.pyplot as plt
import numpy as np

x_epoch = [1,5,10,15,20,25,30,35,40,45,50,55,60]
y_interp_loss = [8.84,3.81,3.00,2.49,2.05,1.79,1.64,1.48,1.44,1.35,1.36,1.31,1.21]
y_conTran_loss = [30.69,22.39,27.07,26.11,23.77,22.68,22.32,21.36,21.76,22.46,22.15,22.35,22.63]

fig,ax = plt.subplots()

plt.xlabel('epoch')
plt.ylabel('loss')

yticks = range(0,35,4)
ax.set_yticks(yticks)

"""set min and max value for axes"""
ax.set_ylim([0,40])
ax.set_xlim([0,70])


plt.plot(x_epoch,y_interp_loss,"r-",label="interpolate")
plt.plot(x_epoch,y_conTran_loss,"b-",label="ConvTranspose2d")

plt.legend(bbox_to_anchor=(1.0,1),loc=1,borderaxespad=0.)

plt.savefig('loss-epoch')
plt.show()


