clear;
close all hidden;

data = chickenpox_dataset;
data = [data{:}];
figure(1), plot(data), title('chickenpox data'); xlabel('month'); ylabel('cases');

numTimeStepsTrain = floor(0.9*numel(data));

data_train = data(1 : numTimeStepsTrain + 1);
data_test = data(numTimeStepsTrain + 1 : end);

mu = mean(data_train);
sigma = std(data_train);

data_train_norm = (data_train - mu) / sigma;
data_test_norm = (data_test - mu) / sigma;

XTrain = data_train_norm(1 : end-1); 
YTrain = data_train_norm(2 : end); 

XTest = data_test_norm(1 : end-1);
numTimeStepsTest = numel(XTest);

% Build LSTM

layers = [sequenceInputLayer(1)
         lstmLayer(100)
         fullyConnectedLayer(1)
         regressionLayer];
options = trainingOptions('adam', 'MaxEpochs', 300, 'GradientThreshold', 1, 'InitialLearnRate', 0.005, ...
    'LearnRateSchedule', 'piecewise', 'LearnRateDropPeriod', 125, 'LearnRateDropFactor', 0.2, 'Verbose', 0, 'Plots', ...
    'training-progress');

net = trainNetwork(XTrain, YTrain, layers, options);
net = predictAndUpdateState(net, XTrain);
[net, YPred] = predictAndUpdateState(net, YTrain(end));

for i=2:numTimeStepsTest
    [net, YPred(:, i)] = predictAndUpdateState(net, YPred(:, i-1), 'ExecutionEnvironment', 'cpu');
end

YPred = sigma * YPred + mu;
YTest = data_test(2: end);

mse = sqrt(mean(YPred - YTest) .^ 2);

figure(2), plot(data_train(1 : end-1)); title('Results of Training'); hold on;
timesteps_number = numTimeStepsTrain : (numTimeStepsTrain + numTimeStepsTest);
plot(timesteps_number, [data(numTimeStepsTrain) YPred]), xlabel('Month'), ylabel('Cases'), ...
    title('Forecast'), legend(['Observed' 'Forecast']);

figure(3),
subplot(211), plot(YTest), title('Forecasting chickenpox cases'); hold on;
plot(YPred,'.-'), legend(['Observed' 'Forecast'])

subplot(212), stem(YPred - YTest) 
title("RMSE = " + mse) 