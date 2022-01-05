clear;
close all hidden;

load abalone_dataset.mat;
load x3.mat;
load xtest3.mat;
load xref3.mat;

% Upload data
inputs = abaloneInputs;
targets = abaloneTargets;
X3 = x3;
Xtest3 = xtest3;
Xref3 = xref3;

% Visualize a part of train dataset
t = [1:100];
for i = 2:8
    plot(t, inputs(i, 1:100)); hold on;
end
figure(1), legend('Length', 'Diameter', 'Height', 'Weight', 'Lost Weight', ...
    'Viscera Weight', 'Sink Weight'), title('Abalone Data');

autoenc = trainAutoencoder(inputs, 5, 'MaxEpochs', 100, 'L2WeightRegularization', 0.02, ...
    'EncoderTransferFunction', 'satlin', 'DecoderTransferFunction', 'purelin', 'SparsityProportion', 0.55);

% Create prediction model
P = predict(autoenc, inputs);
error = mse(inputs - P);
disp(error);

% Visualize estimation of the prediction model
figure(2),
plot(t, inputs(2, 1:100), t, P(2, 1:100)); hold on;
legend('Original Data', 'Predicted Values'), title('Estimation of Prediction');


% Release Sparse Autoencoder
new_autodec = trainAutoencoder(X3, 45, 'MaxEpochs', 500, 'EncoderTransferFunction', 'satlin', 'DecoderTransferFunction', 'purelin', ...
    'L2WeightRegularization', 0.01, 'SparsityProportion', 0.3, 'SparsityRegularization', 4);

% Prediction
new_P = predict(new_autodec, Xtest3);
disp(mse(Xtest3 - new_P));

% Visualize the result
figure(3), plot([1:length(X3)], Xtest3, 'r.', [1:length(X3)], new_P, 'bo', [1:length(X3)], Xref3, 'c.');
legend('Noisy Data', 'Prediction', 'Reference'); title('Sparse Autoencoder');

