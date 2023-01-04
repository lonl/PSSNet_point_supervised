#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

####################6months
#resunet
recall_1 = np.array([0.988665,0.987424,0.970853,0.970368,0.969882,0.969396,0.968586,0.969396,0.968586,0.968586,0.968586])
precision_1 = np.array([0.568639,0.921384,0.970853,0.985528,0.985845,0.98638,0.987074,0.98638,0.987074,0.987074,0.987128])

#faster r-cnn
recall_2 = np.array([0.990831,0.973809,0.971332,0.96954,0.967801,0.965535,0.962901,0.959422,0.955207,0.944456,0])
precision_2 = np.array([0.132153,0.975248,0.980009,0.982065,0.983716,0.984948,0.986396,0.987578,0.989735,0.985579,0.9888])

#Fcn-8
recall_3 = np.array([0.992282,0.950127,0.939116,0.929724,0.922599,0.915313,0.915205,0.915205,0.915205,0.915205,0.915313])
precision_3 = np.array([0.284314,0.953575,0.962654,0.96786,0.970807,0.974318,0.973476,0.973476,0.973476,0.973476,0.974318])


plt.figure()
plt.ylim(0.0,1.0)
plt.xlim(0.5,1.0)


plt.plot(recall_1,precision_1,"r-",label="Ours")
plt.plot(recall_2,precision_2,"--",label="Faster R-CNN")
plt.plot(recall_3,precision_3,"b-.",label="FCN-8")

plt.xlabel('recall')
plt.ylabel('precision')


plt.legend()
plt.savefig('6months-pr')
plt.show()


##########################12 months
#resunet
recall_1= np.array([0.973838,0.982172,0.962943,0.95835,0.961736,0.956637,0.961035,0.956637,0.961035,0.956637,0.960568])
precision_1 = np.array([0.578312,0.852988,0.950621,0.952086,0.953165,0.954334,0.953869,0.954334,0.953869,0.954334,0.953074])

#faster r-cnn
recall_2 = np.array([0.989009,0.977239,0.974603,0.972523,0.971112,0.969516,0.967399,0.965357,0.961421,0.954367,0])
precision_2 = np.array([0.133283,0.934557,0.948883,0.954519,0.958514,0.961837,0.965356,0.968089,0.972142,0.976855,0.99999])

#Fcn-8
recall_3 = np.array([0.978474, 0.887271,0.867341,0.850136,0.833243,0.822655,0.822655,0.819424,0.822655,0.819424,0.819424])
precision_3 = np.array([0.290061,0.904991,0.920325,0.927664,0.9337,0.938621,0.938621,0.938018,0.938621,0.938018,0.938018])


plt.figure()
plt.ylim(0.0,1.0)
plt.xlim(0.5,1.0)


plt.plot(recall_1,precision_1,"r-",label="Ours")
plt.plot(recall_2,precision_2,"--",label="Faster R-CNN")
plt.plot(recall_3,precision_3,"b-.",label="FCN-8")

plt.xlabel('recall')
plt.ylabel('precision')


plt.legend()
plt.savefig('12months-pr')
plt.show()
