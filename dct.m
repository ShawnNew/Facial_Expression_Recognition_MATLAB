%--------------------------------------------------------------------------  
% ����˵����DCT�㷨  
% ���ղ���  
%   dataSet���ݼ�,ά����num*height*weight
%   num
%     
% ��������  
%     dct_feature���Ǿ���DCT������ȡ������������� 
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