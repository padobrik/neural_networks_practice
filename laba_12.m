clear;
close all hidden;

t = [1,-1;
    -1, 1];
figure(1), plot(t(1,:),t(2,:),'r*'), axis([-1.1 1.1 -1.1 1.1]), %отодвинули поля
title('Hopfield Network'), hold on; 
net = newhop(t);
[Y, Pf, Af] = sim(net,2,[],t);
a = {rands(2,1)};
[y, Pf, Af] = sim(net,{1 20},[],a);
record = [cell2mat(a) cell2mat(y)];
start = cell2mat(a);
plot(start(1,1),start(2,1),'b+',record(1,:),record(2,:)), color = 'rgbmy';
for i=1:25
    a = {rands(2,1)};
    [y, Pf, Af] = sim(net,{1 20},[],a);
    record = [cell2mat(a) cell2mat(y)];
    start = cell2mat(a);
    plot(start(1,1),start(2,1),'b+',record(1,:),record(2,:)), color(rem(i,5)+1);
end