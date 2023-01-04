clear;
clc;
d_a = [60,60,40,40,40,30,30];
d_b = [30,20,40,30,20,30,20];
[d_a,d_b]=meshgrid(d_a,d_b);
F = [0.970,0.901,0.979,0.972,0.908,0.971,0.908];
plot3(d_a,d_b,F);
view(3);
xlabel('d_a'),ylabel('d_b'),zlabel('F');
