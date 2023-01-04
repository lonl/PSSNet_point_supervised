### result_compare

* 在相同的阈值范围下对Fast-RCNN 和 LCFCN进行比价：

* ```python
  Condition1:
  comb_xy:dis_comb=60
  compare-xy:dis_compare = 20
      
  LCFCN:    
  06-months:    
  labeled=18527 predicted=33508 True=16705
  Accuracy=:0.901657
  
  12-months:
  labeled=25690 predicted=37764 True=19445
  Accuracy=:0.756909
  
          
  Fast-RCNN:
  06-months:  
  labeled=18976 predicted=36940 True=17964
  Accuracy: 0.946669 
      
  12-months:
  labeled=26932 predicted=42025 True=24799
  Accuracy: 0.920801
  ```

* ```python
  Condition2:
  comb_xy:dis_comb=60
  compare-xy:dis_compare = 30
   
  LCFCN:    
  06-months:    
  labeled=18527 predicted=33508 True=17824
  Accuracy=:0.962055
  
  12-months:
  labeled=25690 predicted=37764 True=23621
  Accuracy=:0.919463
      
  Fast-RCNN:
  06-months:  
  labeled=18976 predicted=36940 True=18422
  Accuracy: 0.970805
  
  12-months:
  labeled=26932 predicted=42025 True=26062
  Accuracy: 0.967696 
   
  ```

* ```python
  Condition3:
  comb_xy:dis_comb=40
  compare-xy:dis_compare = 30
  
  LCFCN:    
  06-months:    
  labeled=18527 predicted=33774 True=17930
  Accuracy=:0.967777
      
  12-months:
  labeled=25690 predicted=38282 True=23731
  Accuracy=:0.923745
     
  Fast-RCNN:
  06-months:  
  labeled=18976 predicted=37776 True=18716
  Accuracy: 0.986298
      
  12-months:
  labeled=26932 predicted=43787 True=26380
  Accuracy: 0.979504
  ```

* ```python
  Condition4:
  comb_xy:dis_comb=40
  compare-xy:dis_compare = 20
  
  LCFCN:    
  06-months:    
  labeled=18527 predicted=33774 True=16899
  Accuracy=:0.912128
  
  12-months:
  labeled=25690 predicted=38282 True=19561
  Accuracy=:0.761425
  
      
  Fast-RCNN:
  06-months:  
  labeled=18976 predicted=37776 True=18538
  Accuracy: 0.976918
  
  12-months:
  labeled=26932 predicted=43787 True=25292
  Accuracy: 0.939106
  ```

* ```python
  Condition5:
  comb_xy:dis_comb=30
  compare-xy:dis_compare = 30
  
  LCFCN:    
  06-months:    
  labeled=18527 predicted=33888 True=17935
  Accuracy=:0.968047
  
  12-months:
  labeled=25690 predicted=38567 True=23746
  Accuracy=:0.924329
      
  Fast-RCNN:
  06-months:  
  labeled=18976 predicted=38338 True=18730
  Accuracy: 0.987036
  
  12-months:
  labeled=26932 predicted=45033 True=26416
  Accuracy: 0.980841
  ```

* ```python
  Condition6:
  comb_xy:dis_comb=30
  compare-xy:dis_compare = 20
  
  LCFCN:    
  06-months:    
  labeled=18527 predicted=33888 True=16912
  Accuracy=:0.91283
  
  12-months:
  labeled=25690 predicted=38567 True=19589
  Accuracy=:0.762515
      
  Fast-RCNN:
  06-months:  
  labeled=18976 predicted=38338 True=18592
  Accuracy: 0.979764
  
  12-months:
  labeled=26932 predicted=45033 True=25412
  Accuracy: 0.943562
  ```

* 结果分析：对于Fast-RCNN来说：当compare-xy相同时，comb_xy精度：30>40>60

* 最好的condition：comb_xy:dis_comb=30
  compare-xy:dis_compare = 30

* 同等条件下，Fast-RCNN 优于 LCFCN

* 60 30 ,40 30 ,30 30 下，对于LCFCN:6月份效果好于12月份。

