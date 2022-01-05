clear;
close all hidden;
load arrhythmia; 

Table=tabulate(categorical(Y));
tTree = templateTree('MinLeafSize',20);
t = templateEnsemble('AdaBoostM1',100,tTree,'LearnRate',0.1);
Mdl = fitcecoc(X,Y,'Learners',t);
view(Mdl.BinaryLearners{1}.Trained{1},'Mode','graph');
L = resubLoss(Mdl,'LossFun','classiferror');