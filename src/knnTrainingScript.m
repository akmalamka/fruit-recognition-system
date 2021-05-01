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