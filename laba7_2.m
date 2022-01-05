clear;
close all hidden;

load fisheriris
t = fitctree(meas,species,'PredictorNames',{'SL' 'SW' 'PL' 'PW'});
view(t);