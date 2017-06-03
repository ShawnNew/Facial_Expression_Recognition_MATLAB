% Solve a Pattern Recognition Problem with a Neural Network
% Script generated by Neural Pattern Recognition app
% Created 13-Mar-2017 21:39:59
%
% This script assumes these variables are defined:
%
%   X - input data.
%   y - target data.
clear; close all; clc;
load('imgdata.mat');

y_m = zeros(length(y), max(y));
for k = 1:length(y)
   y_m(k,:) = zeros(1, max(y));
   y_m(k, y(k)) = 1;
end
t = y_m;

x = H';
t = t';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = 7;
net = patternnet(hiddenLayerSize);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
tInd = tr.testInd;
tstOutputs = net(x(:,tInd));
[a b] = max(tstOutputs, [], 1);
tstPerform = perform(net,y(tInd),b')
% pred = net(x);
% e = gsubtract(t',y);
% performance = perform(net,t',y)
% tind = vec2ind(t');
% yind = vec2ind(y);
% percentErrors = sum(tind ~= yind)/numel(tind);

% View the Network
% view(net)

% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, ploterrhist(e)
% figure, plotconfusion(t',y)
% figure, plotroc(t',y)

