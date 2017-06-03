function picture = cut(img)
% cut the input image into a square to fit the neural network
% the input img is in whichever size, and we should could the redundent
% area.
scaleX = size(img, 2);
scaleY = size(img, 1);
if scaleX > scaleY
    picture = img(:,...
        (scaleX - scaleY)/2 + 1 : (scaleX - (scaleX - scaleY)/2));
else
    picture = img(...
        (scaleY - scaleX)/2 + 1 : (scaleY - (scaleY - scaleX)/2),:);
end