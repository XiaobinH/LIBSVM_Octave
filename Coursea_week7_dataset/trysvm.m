%% Initialization
clear ; close all; clc

%% PART 1: load and visualize ex6data1
fprintf('Loading and Visualizing Data ...\n')
% Load from ex6data1: 
% You will have X, y in your environment
load('ex6data1.mat');
y=(y-(y==zeros(size(y))));
plotsvm(X, y)
fprintf('load and visualize ex6data1 : done. Press to continue\n');
pause;

%% PART 2: train model with SVM on ex6data1, small data set
% Linear SVM
fprintf('\nTraining Linear SVM ...\n')
model = svmtrain(y, X, '-t 0');
svmvisualizeBoundary(X, y, model);
[predict_label, accuracy, prob_estimates] = svmpredict(y, X, model);
fprintf('Linear SVM : done. Press to continue\n');
pause;
fprintf('\nTraining polynomial SVM ...\n')
model = svmtrain(y, X, '-t 1');
svmvisualizeBoundary(X, y, model);
[predict_label, accuracy, prob_estimates] = svmpredict(y, X, model);
fprintf('polynomial SVM : done. Press to continue\n');
pause;
fprintf('\nTraining gaussian SVM ...\n')
model = svmtrain(y, X, '-t 2');
svmvisualizeBoundary(X, y, model);
[predict_label, accuracy, prob_estimates] = svmpredict(y, X, model);
fprintf('gaussian SVM : done. Press to continue\n');
pause;
fprintf('\nTraining sigmoid SVM ...\n')
model = svmtrain(y, X, '-t 3');
svmvisualizeBoundary(X, y, model);
[predict_label, accuracy, prob_estimates] = svmpredict(y, X, model);
fprintf('Under sigmoid SVM, all points are predicted for the same results. weird.\n');
fprintf('sigmoid SVM : done. Press to continue\n');
pause;

%% PART 3: load and visualize ex6data2
fprintf('Loading and Visualizing Data ...\n')
% Load from ex6data2: 
% You will have X, y in your environment
load('ex6data2.mat');
y=(y-(y==zeros(size(y))));
plotsvm(X, y)
fprintf('load and visualize ex6data2 : done. Press to continue\n');
pause;

%% PART 4: train model with SVM on ex6data2, larger data set
fprintf('\nTraining Linear SVM ...\n')
model = svmtrain(y, X, '-t 0');
svmvisualizeBoundary(X, y, model);
[predict_label, accuracy, prob_estimates] = svmpredict(y, X, model);
fprintf('Linear SVM : done. Press to continue\n');
pause;
fprintf('\nTraining polynomial SVM ...\n')
model = svmtrain(y, X, '-t 1');
svmvisualizeBoundary(X, y, model);
[predict_label, accuracy, prob_estimates] = svmpredict(y, X, model);
fprintf('polynomial SVM : done. Press to continue\n');
pause;
fprintf('\nTraining gaussian SVM ...\n')
model = svmtrain(y, X, '-t 2 -g 100');
svmvisualizeBoundary(X, y, model);
[predict_label, accuracy, prob_estimates] = svmpredict(y, X, model);
fprintf('gaussian SVM : done. Press to continue\n');
pause;
fprintf('\nTraining sigmoid SVM ...\n')
model = svmtrain(y, X, '-t 3');
svmvisualizeBoundary(X, y, model);
[predict_label, accuracy, prob_estimates] = svmpredict(y, X, model);
fprintf('Under sigmoid SVM, all points are predicted for the same results. weird.\n');
fprintf('sigmoid SVM : done. Press to continue\n');
pause;








