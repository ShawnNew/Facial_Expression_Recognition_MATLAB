% Read the facial expression image into our workspace.
% After read the image from the files, then use the LBP method to extract
% the featuers from the images.
% Use 'save' function to save the variable into .mat file

%Set ys as the labels of the expressions.
%Set:
%     AN to 1,
%     DI to 2,
%     FE to 3,
%     HA to 4,
%     NE to 5,
%     SA to 6,
%     SU to 7.

clear;
img_list = dir(fullfile('JAFEE_Database', 'jaffe', '*.tiff'));    %create a list of image
X = zeros(size(img_list, 1), 256*256);
y = zeros(size(img_list));
an_counter = 0;
di_counter = 0;fe_counter = 0;ha_counter = 0;ne_counter = 0;
sa_counter = 0;su_counter = 0;
% mapping=getmapping(8,'u2');%lbp mapping list
% X = zeros(size(img_list, 1), (8*(8-1)+3)*16*16);  % The number of features: numCells*(P(P-1)+3)

for i = 1:size(img_list,1)
    a = imread(['JAFEE_Database/jaffe/',img_list(i).name]);  %read the image file
    c = reshape(a, 1, 256*256);
%     lbpc = extractLBPFeatures(c, 'CellSize', [16 16]);
%     MappedData = mapminmax(lbpc, 0, 0.5);  %map the data to [0, 0.5]
    
%     img_temp = reshape(MappedData, 1,...
%         size(MappedData, 1) * size(MappedData, 2));  %convert the image into vector
%     X(i, :) = MappedData;
    X(i, :) = c;
    
    %compute the y's value
%     an_counter = 0;
%     di_counter = 0;fe_counter = 0;ha_counter = 0;ne_counter = 0;
%     sa_counter = 0;su_counter = 0;
    switch  img_list(i).name(4:5)
        case 'AN'
            y(i) = 1;
            an_counter = an_counter + 1;
        case 'DI'
            y(i) = 2;
            di_counter = di_counter + 1;
        case 'FE'
            y(i) = 3;
            fe_counter = fe_counter + 1;
        case 'HA'
            y(i) = 4;
            ha_counter = ha_counter + 1;
        case 'NE'
            y(i) = 5;
            ne_counter = ne_counter + 1;
        case 'SA'
            y(i) = 6;
            sa_counter = sa_counter + 1;
        otherwise
            y(i) = 7;
            su_counter = su_counter + 1;
    end
end
y_m = zeros(size(y, 1), max(y));
for k = 1:size(y, 1)
   y_m(k,:) = zeros(1, max(y));
   y_m(k, y(k)) = 1;
end
y = y_m;

save('imgdata.mat','X','y');





