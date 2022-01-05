clear;
close all hidden;

path = fullfile(matlabroot, 'toolbox', 'nnet', 'nndemos', 'nndatasets', 'DigitDataset');

images = imageDatastore(path, 'IncludeSubfolders',true,'LabelSource','foldernames');
perm = randperm(10000, 20);
for i = 1:20
    subplot(5,4,i);
    imshow(images.Files{perm(i)});
end

label = countEachLabel(images);
img = readimage(images, 1);
image = readimage(images, i);
size(image)
numTrainFiles = 750;
[imagesTrain, imagesValidate] = splitEachLabel(images, numTrainFiles, 'randomize');

layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(10)

    softmaxLayer

    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imagesValidate, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(imagesTrain,layers,options);
YPred = classify(net,imagesValidate);
YValidation = imagesValidate.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);