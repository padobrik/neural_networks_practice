clear;
close all hidden;
load Pnorm;
load Tnorm;
load x;
load y;
load z;
P = x;
T = y;
K = z;

%Обучение MLP с использованием метода второго порядка
net3 = newff(minmax(Pn), [10,10,1], {'tansig','logsig', 'purelin'}, 'trainlm');
net3.trainParam.show = 20;
net3.trainParam.epochs = 200;
net3.trainParam.goal = 1e-3;
net3 = train(net3,Pn,Tn);
S = sim(net3,Pn);
a = poststd(S, meanT, stdT); %разнормирование
deltaMass = abs(a-T)./max(T); 
delta = mean(deltaMass);
f = figure(1);
clf; 
[N, M] = size(x); 
plot(1:M, T, 1:M, a), legend('target', 'NetOutput'), grid,
set(f, 'Position', [80 50 600 480]),
set(get(f, 'CurrentAxes'), 'Position', [0.13 0.11 0.775 0.785]),
xlabel('Sample'), ylabel('NN output'), title('Train Result');
gtext({'NetTrain error = ', num2str(delta)});


PTestN = trastd(x, meanP, stdP);
aNTest = sim(net3, PTestN);
aTest = poststd(aNTest, meanT, stdT);
[L, R] = size(x);
figure(2);
plot(1:R, aTest, 1:R, z), grid; title('Test result'),
legend('Test', 'NetOutput');
deltaMass2 = abs(aTest - z)./max(z);
delta2 = mean(deltaMass2);
gtext({'Test error = ', num2str(delta2)});



