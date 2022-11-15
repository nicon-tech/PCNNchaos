clear all, close all, clc
%% Add appropriate folders to path
%addpath('./data');
%% import data
X = csvread('lorenzExample.csv',0,0); %data
%% build training data
ind_train = length(X)*0.7;
Xtrain = X(1:ind_train,:);
writematrix(Xtrain,'lorenzExample_train1_x.csv')
%% build test data
ind_test = length(X)*0.1;
ind_val = length(X)*0.2;
Xtest = X(ind_train+1:end-ind_val,:);
writematrix(Xtest,'lorenzExample_test_x.csv')
%% build validation data
Xval = X(end-ind_val+1:end,:);
writematrix(Xval,'lorenzExample_val_x.csv')
%% plot
plot3(X(:,1),X(:,2),X(:,3))