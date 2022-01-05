clear;
close all hidden;
load Plearn;
load Ptest;
load Tlearn;
load Ttest;
tr=100; %кол-во эпох
%Pseq=con2seq(Plearn);
%Tseq=con2seq(Tlearn);
net=newelm([-2 2], [10 1], {'tansig', 'purelin'}, 'trainlm');
net.TrainParam.epochs=280;
net.TrainParam.show=10;
net.TrainParam.goal=1e-10;
[net,tr] = train(net,Plearn,Tlearn);
a = sim(net,Plearn);
time=1:length(Plearn);
figure(1), plot(time, Tlearn, time, a);