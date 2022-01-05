clear;
close all hidden;
load x;
load y;
load z;
P = x;
T = y;
K = z;
%производим предварительную обработку данных, чтобы среднее значение было 0, а стандартное отклонение 1
[Pn, meanP, stdP, Tn, meanT, stdT] = prestd(P, T); 
Pnorm = [Pn, meanP, stdP]; %объединяем в массив
Tnorm = [Tn, meanT, stdT];
%сохраняем обработанные значения, чтобы для дргуих методов не производить обработку заново
save Pnorm;
save Tnorm;

%Обучение MLP с использованием градиентного метода
net = newff(minmax(Pn), [10,10,1], {'tansig','logsig', 'purelin'}, 'trainrp');
net.trainParam.show = 65;
net.trainParam.epochs = 200;
net.trainParam.goal = 1e-3;
net = train(net,Pn,Tn);
S = sim(net,Pn);
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


PTestN = trastd(x, meanP, stdP); %обрабатываем данные уже посчитанных значений
aNTest = sim(net, PTestN); %симулируем
aTest = poststd(aNTest, meanT, stdT); %разнормирование
[L, R] = size(x);
figure(2);
plot(1:R, aTest, 1:R, z), grid; title('Test result'),
legend('Test', 'NetOutput');
%вычисляем ошибку
deltaMass2 = abs(aTest - z)./max(z);
delta2 = mean(deltaMass2);
gtext({'Test error = ', num2str(delta2)});



