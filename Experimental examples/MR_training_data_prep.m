clc, clear; close all;
% Unit cm N
load('mrdamper.mat');
u=V(1:2001); y_ref=F(1:2001);
clear Ts V F;
save('MR_training_data.mat','u','y_ref'); %,'-double'
