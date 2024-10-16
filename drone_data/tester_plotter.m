
clear;
clc;

%tester plotter
load("result.mat");
% load("C:\Users\STAJYER\Desktop\data_staj\snr_18\pixs_5_9\binary\snr_18pixes5_9_v1bnr.mat");
% load("C:\Users\STAJYER\Desktop\data_staj\snr_18\pixs_5_9\specs\snr_18pixes5_9_v1.mat");
% pred=squeeze(pred);
figure;
%first plot
ax1 = subplot(1,2,1);
imagesc(input);
% imagesc(P);

ax2= subplot(1,2,2);
pred=squeeze(pred);
imagesc(pred);
% imagesc(p);

linkaxes([ax1,ax2])