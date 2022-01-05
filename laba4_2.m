clear;
close all hidden;
P=[2.5 2.5; 0.5 0.5; 0.5 2.5; 2.5 0.5; 1 1; 2 1; 2 2; 1 2; 1.5 1.5; 1.7 1.3; 1.3 1.7; 1.3 1.5]';
Tc=[1 1 1 1 2 2 2 2 3 3 3 3];
figure(1); 
% нарисуем каждую точку на плоскости своим цветом, разный цвет для разных классов
hold on;
for i=1:12
    if(Tc(i)==1)
        plot(P(1,i),P(2,i),'.r','markersize',30); end; hold on;
    if(Tc(i)==2)
        plot(P(1,i),P(2,i),'.g','markersize',30); end; hold on;
    if(Tc(i)==3)
        plot(P(1,i),P(2,i),'.b','markersize',30); end; hold on;
end
axis([0 3 0 3]);
title('Three vector and their classes'), %три точки означает всё
xlabel('P(1,:)'); ylabel('P(2,:)');
%----------------------------------------------------------------------
%теперь делаем сеть PNN
T=ind2vec(Tc);
spread=0.3;
net=newpnn(P,T,spread);
A=sim(net,P); %здесь лежит результат
%так как мы преобразовываали Т, то теперь нужно его преобразовать обратно
Ac=vec2ind(A);
%----------------------------------------------------------------------
for i=1:12
    if(Tc(i)==1)
        plot(P(1,i),P(2,i),'.r','markersize',30); end; hold on;
    if(Tc(i)==2)
        plot(P(1,i),P(2,i),'.g','markersize',30); end; hold on;
    if(Tc(i)==3)
        plot(P(1,i),P(2,i),'.b','markersize',30); end; hold on;
end
axis([0 3 0 3]);
title('Testing the network'), 
xlabel('P(1,:)'); ylabel('P(2,:)');
%----------------------------------------------------------------------
%чтобы не транспонировать, сразу сделаем столбцом
p=[2;1.5];
a=sim(net,p);
%получаем число, которое нужно преобразовать в индекс
ac=vec2ind(a); hold on; 
%теперь нужно нанести на график
plot(p(1),p(2),'.','markersize',30, 'color', [1 1 0]); %цвет - свойство, которое называется color, поэтому цвет можно ввести как RGB. Здесь как раз показываем это свойство в действии. 
hold off;
%----------------------------------------------------------------------
p1=0:0.05:3;
p2=p1;
[P1,P2]=meshgrid(p1,p2);
pp=[P1(:),P2(:)]';
aa=sim(net,pp);
aa=full(aa);

m=mesh(P1,P2,reshape(aa(1,:),length(p1),length(p2))); %это трехмерный рисунок
set(m,'facecolor',[0 .5 .7],'linestyle','none');
hold on;
m=mesh(P1,P2,reshape(aa(2,:),length(p1),length(p2))); %это трехмерный рисунок
set(m,'facecolor',[0.7 .8 .1],'linestyle','none');
hold on;
m=mesh(P1,P2,reshape(aa(3,:),length(p1),length(p2))); %это трехмерный рисунок
set(m,'facecolor',[0 .7 .9],'linestyle','none');
%----------------------------------------------------------------------
hold on;
for i=1:12
    if(Tc(i)==1)
        plot3(P(1,i),P(2,i),1.1,'.','markersize',30, 'color', [1 0 0]); end; hold on; %1.1 отступ вверх и plot3 тоже трехмерный график
    if(Tc(i)==2)
        plot3(P(1,i),P(2,i),1.1,'.','markersize',30, 'color', [0 1 0]); end; hold on;
    if(Tc(i)==3)
        plot3(P(1,i),P(2,i),1.1,'.','markersize',30, 'color', [0 0 1]); end; hold on;
end
