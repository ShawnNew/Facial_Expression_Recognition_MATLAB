%--------------------------------------------------------------------------  
% 函数说明：DCT算法  
% 接收参数  
%   dataSet数据集,维数是num*height*weight
%   num
%     
% 函数返回  
%     dct_feature，是经过DCT特征提取后的人脸特征集 
%   
 
%--------------------------------------------------------------------------  

function dct_feature = dct(dataSet, num)
    
    m = size(dataSet, 1);
    dct_feature = zeros(m, num);
    for i = 1:m
       temp = dct2(dataSet(i, :, :));
%        temp = reshape(F, size(F,2), size(F,3));
       dct_feature(m, :) = temp(1:num);
    end

end