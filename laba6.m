clear;
close all hidden;

%здесь делаем при помощи регрессии
%Создаем несколько узлов со случайными координатами на плоскости
angles = 0:0.2:4*pi;
X = [sin(angles); cos(angles)];
X=X.*(0.5*angles+0.1);
figure(1), plot(X(1,:),X(2,:),'+r'), title('Узлы заданные регрессией'); %вывели смоделированные данные для узлов
net=selforgmap([55 2]); %строим самоорганизующую карту, на вход пишем размер вектора-строки. То есть эта функция перебираем все возможные комбинации и ищет наикрочайшее расстояние...
%межжду узлами и моделирует это пространство
%подбираем этот размер так, чтобы она описывала то распределение
net=configure(net,X); % позволяет подготовить сеть к данным перед обучением
[net,tr]=train(net,X); 
figure(2), plotsompos(net); %после тренировки нужно её вывести. выведутся веса каждого нейрона сети вместе со связями между соседними нейронами

%теперь сделаем тоже самое, но при помощи градиентного спуска
clear;
angles = 0:0.2:4*pi;
XX = rand(2,1000);
figure(3), plot(XX(1,:),XX(2,:),'+g'), title('Узлы заданные градиентным спуском');
net2=selforgmap([55 2]);
%net2=configure(net2,XX);
[net2,tr2]=train(net2,XX); 
figure(4), plotsompos(net2);

%теперь сделаем классификацию по входным данным
clear;
I = iris_dataset;
size(I)
net3 = selforgmap([8 8]);
[net3,tr3]=train(net3,I);
figure(5), plotsompos(net3);
figure(6), plotsomhits(net3,I);
figure(7), plotsomplanes(net3);
figure(8), plotsomnc(net3,I);
figure(9), plotsomnd(net3,I);
figure (10), plotsompos (net3,I);


