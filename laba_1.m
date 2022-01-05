clear
close all hidden;

% Define logical sentence
P = [0 0 0 0 1 1 1 1; 0 0 1 1 0 0 1 1; 0 1 0 1 0 1 0 1];
T = [0 1 1 0 0 0 0 1];

% Train simple perceptron
net = newp(minmax(P), 1); 
net.trainParam.epochs = 100;
net = train(net, P, T);

% Do simulation
Y = sim(net,P); 

% Plot results
figure(1), plotpv(P, T); 
figure(2), hold on, plotpv(P, Y);
plotpc(net.IW{1}, net.b{1});

% Train two-layer perceptron
netn = newff(minmax(P), [2, 1], {'tansig', 'logsig'});
net = train(net, P, T);
Yn = sim(net, P);

% Plot results for this neural network
figure(3), plotpv(P,double(T));
figure(4), hold on, plotpv(P, double(Yn)), plotpc(netn.IW{1}, netn.b{1});