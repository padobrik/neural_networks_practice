clear;
close all hidden;

r1 = sqrt(rand(100, 1));
t1 = 2 * pi * rand(100, 1);
data1 = [r1 .* cos(t1), r1 .* sin(t1)]; % first class


r2 = sqrt(3 * rand(100, 1) + 1);
t2 = t1;
data2 = [r2 .* cos(t2), r2 .* sin(t2)]; % second class

figure(1);
plot(data1(:, 1), data1(:, 2), 'r.', 'MarkerSize', 10);
hold on;
plot(data2(:, 1), data2(:, 2), 'b.', 'MarkerSize', 10);

ezpolar(@(x)1); % to polar coordinates
ezpolar(@(x)2);
axis equal;
hold off;

% generating train set
data3 = [data1; data2];

zClass = ones(200, 1);
zClass(1:100) = -1;

% train SVM

SVMModel = fitcsvm(data3, zClass, 'KernelFunction', 'rbf', 'BoxConstraint', inf, 'ClassNames', [-1 1]);
d = 0.02;
[x1grid, x2grid] = meshgrid(min(data3(:, 1)):d:max(data3(:, 1)), ...,
    min(data3(:, 2)):d:max(data3(:, 2)));
xgrid = [x1grid(:), x2grid(:)];

[~, score] = predict(SVMModel, xgrid);

figure(2); 
h(1:2) = gscatter(data3(:, 1), data3(:, 2), zClass, 'rb', '.');
hold on;

ezpolar(@(x)1);

h3 = plot(data3(SVMModel.IsSupportVector, 1), data3(SVMModel.IsSupportVector, 2), 'go');
contour(x1grid, x2grid, reshape(score(:, 2), size(x1grid)), [0 0], 'm');
legend(h, {'-1', '+1'});
axis equal;
hold off;