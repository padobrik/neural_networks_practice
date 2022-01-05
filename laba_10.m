clear;
close all hidden;
load carsmall;
rng default;

X = [Horsepower, Weight];
Y = MPG;
figure(1),
subplot(121), plot(X(:, 1), Y, 'b.'), grid; hold on
subplot(122), plot(X(:, 2), Y, 'r.'), grid; hold on

% creating only standartized model
Mdl = fitrsvm(X, Y, 'Standardize', true);
Mdl.ConvergenceInfo.Converged;

std = resubLoss(Mdl); % standart deviation
yfit = resubPredict(Mdl); % output

figure(2),
subplot(121), plot(X(:, 1), Y, 'b+'), grid; hold on
subplot(122), plot(X(:, 2), Y, 'r+'), grid; hold on
subplot(121), plot(Mdl.X(:, 1), yfit, '.k'), xlabel('Horsepower'), ylabel('MPG');
subplot(122), plot(Mdl.X(:, 2), yfit, '.g'), xlabel('Weight'), ylabel('MPG');
for i = 1:length(yfit(:))
    if (Mdl.IsSupportVector(i) == 1)
        subplot(121), plot(Mdl.X(i, 1), yfit(i), 'ok'), xlabel('Horsepower'), ylabel('MPG');
        subplot(122), plot(Mdl.X(i, 2), yfit(i), 'og'), xlabel('Horsepower'), ylabel('MPG');
    end
end

% creating standartized & KFold CV
Mdl2 = fitrsvm(X, Y, 'Standardize', true, 'KFold', 2);
std2 = kfoldLoss(Mdl2);
yfit2 = kfoldPredict(Mdl2);
figure(3),
subplot(121), plot(X(:, 1), Y, 'b+'), grid; hold on
subplot(122), plot(X(:, 2), Y, 'r+'), grid; hold on
subplot(121), plot(Mdl2.X(:, 1), yfit2, '.k'), xlabel('Horsepower'), ylabel('MPG');
subplot(122), plot(Mdl2.X(:, 2), yfit2, '.g'), xlabel('Weight'), ylabel('MPG');
for i = 1:length(yfit(:))
    if (Mdl2.IsSupportVector(i) == 1)
        subplot(121), plot(Mdl2.X(i, 1), yfit2(i), 'ok'), xlabel('Horsepower'), ylabel('MPG');
        subplot(122), plot(Mdl2.X(i, 2), yfit2(i), 'og'), xlabel('Horsepower'), ylabel('MPG');
    end
end

% Standartized & KFold CV & Kernel
Mdl3 = fitrsvm(X, Y, 'Standardize', true, 'KFold', 5, 'KernelFunction', 'gaussian');
std3 = kfoldLoss(Mdl3);
yfit3 = kfoldPredict(Mdl3);
figure(4),
subplot(121), plot(X(:, 1), Y, 'b+'), grid; hold on
subplot(122), plot(X(:, 2), Y, 'r+'), grid; hold on
subplot(121), plot(Mdl3.X(:, 1), yfit3, '.k'), xlabel('Horsepower'), ylabel('MPG');
subplot(122), plot(Mdl3.X(:, 2), yfit3, '.g'), xlabel('Weight'), ylabel('MPG');
for i = 1:length(Mdl3.X(:, 1))
    if (Mdl3.IsSupportVector(i) == true)
        subplot(121), plot(Mdl3.X(:, 1), yfit3, 'ok'), xlabel('Horsepower'), ylabel('MPG');
        subplot(122), plot(Mdl3.X(:, 2), yfit3, 'og'), xlabel('Horsepower'), ylabel('MPG');
    end
end

% -//- + Optimized
Mdl4 = fitrsvm(X, Y, 'Standardize', true, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName','expected-improvement-plus'));
std4 = resubLoss(Mdl4);
yfit4 = resubPredict(Mdl4);
figure(5),
subplot(121), plot(X(:, 1), Y, 'b+'), grid; hold on
subplot(122), plot(X(:, 2), Y, 'r+'), grid; hold on
subplot(121), plot(Mdl4.X(:, 1), yfit4, '.k'), xlabel('Horsepower'), ylabel('MPG');
subplot(122), plot(Mdl4.X(:, 2), yfit4, '.g'), xlabel('Weight'), ylabel('MPG');
for i = 1:length(yfit(:, 1))
    if (Mdl4.IsSupportVector(i) == 1)
        subplot(121), plot(Mdl.X(i, 1), yfit4(i), 'ok'), xlabel('Horsepower'), ylabel('MPG');
        subplot(122), plot(Mdl.X(i, 2), yfit4(i), 'og'), xlabel('Horsepower'), ylabel('MPG');
    end
end