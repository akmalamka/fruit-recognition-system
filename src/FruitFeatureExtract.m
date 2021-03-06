function [ featureVector ] = FruitFeatureExtract( image )

%Read in the image
% figure,imshow(image);
% title('Original image')

%adjust image to accentuate colors
img = imadjust(image,[.2 .2 .2; .65 .65 .65],[]);

%Convert image to HSV to be able to take Saturation channel
img = rgb2hsv(img);

% level = graythresh(img);
% bw_img = im2bw(img,level);
% bw_img = imclose(bw_img,ones([3,3]));

%grab the seperate hue, saturation, and value channels
hue = img(:,:,1);
saturation = img(:,:,2);
value = img(:,:,3);
[row, col] = size(saturation);

% threshold the saturation channel of the image
threshold_sat = saturation > 0.4;

%convert from logical to double
thresh = +threshold_sat;

%perform close to close holes and then fill in any extra holes
for u = 1:5
    thresh = imclose(thresh,ones(9));
end
thresh = imfill(thresh,'holes');

%find the connected components and take out anything less than 1000 pixels because it is noise and not fruit
connCompThreshold = 1000;
CC = bwconncomp(thresh);
 
for i = 1:CC.NumObjects
    L = length(CC.PixelIdxList{i});
    if L < connCompThreshold
        thresh(CC.PixelIdxList{i}) = 0;
    end
end

%find the fruit in the image
CC = bwconncomp(thresh);

% Extract the fruit from the image
temp = zeros(row,col);
temp(CC.PixelIdxList{1}) = 1;
% figure, imshow(temp)
% title('Indiviual Fruit that was extracted out')

%Use regionprops to get the bounding box to take out the image and other feature of the fruit
stats = regionprops(temp,'Area','Perimeter','BoundingBox','Eccentricity','Centroid','FilledImage');

%get individual parts of BoundingBox to get X, Y, Width, Height
%x is the leftmost pixel for the CC
x = stats.BoundingBox(1); x = round(x);
%y is the topmost pixel for the CC
y = stats.BoundingBox(2); y = round(y);
%width is the number of pixels from x to the right
width = stats.BoundingBox(3); width = round(width);
%width is the number of pixels from y to the bottom of the image
height = stats.BoundingBox(4); height = round(height);
fruit_size = [width,height];

%get short to longer width and height vector to get rid of a few orientation problem
[T,I_max] = max(fruit_size);
[T,I_min] = min(fruit_size);
longer = fruit_size(I_max);
shorter = fruit_size(I_min);

%crop out the sub image to send to knn and feature selection
subImage = image(y:(y + height -1),x:(x + width -1),:);

%Threshold the unneccesarry parts in the RGB to white
filledImage = stats.FilledImage; 
bIndinces = find(filledImage == 0);
%seperate out each channel in the rgb and set it equal to white
channelR = subImage(:,:,1);
channelG = subImage(:,:,2);
channelB = subImage(:,:,3);
channelR(bIndinces) = 255;
channelG(bIndinces) = 255;
channelB(bIndinces) = 255;
rgbOut = cat(3, channelR, channelG, channelB);

%Detect the color of the cropped image
[ clr ] = findFeat(rgbOut);

%Return the feature vector of the image
featureVector = [stats.Eccentricity, longer/1000, shorter/1000, clr/3];
end

