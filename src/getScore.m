oBaseDir = '../FruitsDB/Oranges/Test/';
aBaseDir = '../FruitsDB/Apples/Test/';
mBaseDir = '../FruitsDB/Mangoes/Test/';

oFiles = dir('../FruitsDB/Oranges/Test/*.jpg');
aFiles = dir('../FruitsDB/Apples/Test/*.jpg');
mFiles = dir('../FruitsDB/Mangoes/Test/*.jpg');

load('featureVectors.mat');
mdl = fitcknn(vector,Y,'NumNeighbors',4,'Standardize',1);

cm = zeros(3,3);

for i = 1:length(oFiles)
    image = imread(strcat(oBaseDir,oFiles(i).name));
    fruitfeature = FruitFeatureExtract(image);
    [label,POSTERIOR, score] = predict(mdl,fruitfeature);
    switch label
       case 'M'
           cm(1,1) = cm(1,1) + 1;
       case 'A'
           cm(1,2) = cm(1,2) + 1;
       case 'O'
           cm(1,3) = cm(1,3) + 1;
    end
end

for i = 1:length(aFiles)
    image = imread(strcat(aBaseDir,aFiles(i).name));
    fruitfeature = FruitFeatureExtract(image);
    [label,POSTERIOR, score] = predict(mdl,fruitfeature);
    switch label
       case 'M'
           cm(2,1) = cm(2,1) + 1;
       case 'A'
           cm(2,2) = cm(2,2) + 1;
       case 'O'
           cm(2,3) = cm(2,3) + 1;
    end
end

for i = 1:length(mFiles)
    image = imread(strcat(mBaseDir,mFiles(i).name));
    fruitfeature = FruitFeatureExtract(image);
    [label,POSTERIOR, score] = predict(mdl,fruitfeature);
    switch label
       case 'M'
           cm(3,1) = cm(3,1) + 1;
       case 'A'
           cm(3,2) = cm(3,2) + 1;
       case 'O'
           cm(3,3) = cm(3,3) + 1;
    end
end

disp(cm)