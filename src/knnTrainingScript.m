close all
clc

%initialize filter
vector = zeros(1,4);

oBaseDir = '../FruitsDB/Oranges/Train/';
aBaseDir = '../FruitsDB/Apples/Train/';
mBaseDir = '../FruitsDB/Mangoes/Train/';

oFiles = dir('../FruitsDB/Oranges/Train/*.jpg');
aFiles = dir('../FruitsDB/Apples/Train/*.jpg');
mFiles = dir('../FruitsDB/Mangoes/Train/*.jpg');
ptr = 1;

%train Oranges
for i = ptr:ptr+length(oFiles)-1
    image = imread(strcat(oBaseDir,oFiles(i-ptr+1).name));
    [ featureVector ] = FruitFeatureExtract( image );
    vector(i,:) = featureVector;
    Y(i,:) = 'O';
end
ptr = ptr + length(oFiles) + 1;

%train Apples
for i = ptr:ptr+length(aFiles)-1
    image = imread(strcat(aBaseDir,aFiles(i-ptr+1).name));
    [ featureVector ] = FruitFeatureExtract( image );
    vector(i,:) = featureVector;
    Y(i,:) = 'A';
end
ptr = ptr + length(aFiles) + 1;

%train Mangoes
for i = ptr:ptr+length(mFiles)-1
    image = imread(strcat(mBaseDir,mFiles(i-ptr+1).name));
    [ featureVector ] = FruitFeatureExtract( image );
    vector(i,:) = featureVector;
    Y(i,:) = 'M';
end
save('featureVectors.mat', 'vector', 'Y');

%% 
clear all
close all
clc

load('featureVectors.mat');
% x - Predictor values, specified as a numeric matrix. Each column of X 
% represents one variable, and each row represents one observation.
% y - Classification values, specified as a numeric vector, categorical vector,
% logical vector, character array, or cell array of strings, with the same
% number of rows as X. Each row of y represents the classification of the 
% corresponding row of X.
%mdl = fitcknn(vector,Y,'NumNeighbors',3,'Standardize',1);
%mdl = ClassificationKNN.fit(vector,Y,'NumNeighbors',3);

% returns a matrix of scores, indicating the likelihood that a label comes 
% from a particular class.
%[label,POSTERIOR, score] = ClassificationKNN.predict(mdl,Xnew);