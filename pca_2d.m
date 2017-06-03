%--------------------------------------------------------------------------  
% 函数说明：2DPCA算法  
% 接收参数  
%   trainingSet:训练数据集，num*height*width的矩阵    
%   numClass:训练样本和测试样本所拥有的类别数  
%   thresthold:阈值，决定特征向量个数的选择，既决定d取值，  
%              若传递的thresthold为小数，如0.95，自动根据阈值计算，d,q的取值；  
%              若传递的thresthold为>=1的整数，30,则会设d=30,numShape=d  
%     
% 函数返回  
%     allprojectionFace，是经过特征提取后的人脸特征集 
%   
 
%--------------------------------------------------------------------------  
  
function allprojectionFace = pca_2d(trainingSet, thresthold)  
     
    numTrainInstance = size(trainingSet,1); %训练样本数  
%     numTestInstance = size(testingSet,1);   %测试样本数  
    height = size(trainingSet,2);          %图像高度  
    width = size(trainingSet,3);         %图像宽度  
%     perClassTrainLen = numTrainInstance/numClass;%每个类别的训练样本数  
%     perClassTestLen = numTestInstance/numClass;%每个类别的测试样本数  
     
    numShape = 5;%训练的时候投影到多少维向量上，numShape<height  
    projection = zeros(width,numShape);  
    allprojectionFace = zeros(numTrainInstance, height, numShape);  
      
      
      
    %以下为训练  
    meanFace = mean(trainingSet,1);  
    %求协方差矩阵GT  
    CovG = zeros(width,width);  
    for n=1:numTrainInstance  
        disToMean = trainingSet(n, :, :)-meanFace;
        disToMean = reshape(disToMean, size(disToMean, 2), size(disToMean, 3));
        CovG = CovG + disToMean' * disToMean;  
    end  
    CovG = CovG/numTrainInstance;  
  
    %求特征值和特征向量  
    [eigenFace,eigenValue] = svd(CovG);  
      
    if(nargin==4)  
        if(thresthold<1)%如果设置阈值为<1认为需要动态获得d的取值  
            %disp('u set thresthold');  
            sumEigenValue = 0;  
            tmp = 0;  
            for xi=1:size(eigenValue,2)  
                sumEigenValue = sumEigenValue+eigenValue(xi,xi);  
            end  
            for xi=1:size(eigenValue,2)  
                tmp = tmp+eigenValue(xi,xi);  
                if(tmp/sumEigenValue>thresthold)  
                    break;  
                end  
            end;  
            numShape = xi;    
        else %否则认为设置的阈值为d的值，即投影到多少维上  
            %disp('u set the d value...');  
            numShape = thresthold;  
        end  
              
        projection = zeros(width,numShape);  
        allprojectionFace = zeros(numTrainInstance, height,numShape);  
      
    end  
  
    %求投影向量projection,即图像的特征矩阵或特征图像  
    for k=1:numShape  
        projection(:,k) = eigenFace(:,k);  
    end  
  
    %保存每个训练样本投影后的特征  
    for inum = 1:numTrainInstance  
        allprojectionFace(inum, :, :)= reshape(trainingSet(inum,:,:),...
            size(trainingSet,2),size(trainingSet,3)) * projection;  
    end;  
      
%     %以下为测试  
%     right = 0;  
%      
%     for x=1:numTestInstance  
%         afterProjection = testingSet(:,:,x)*projection;     
%         error = zeros(numTrainInstance,1);  
%         for i=1:numTrainInstance  
%             %计算重构图像矩阵到各个类别图像矩阵间的距离  
%             miss = afterProjection -allprojectionFace(:,:,i);  
%             for j=1:size(miss,2)  
%                 error(i) =error(i)+ norm(miss(:,j));  
%             end  
%         end;  
%          
%         [errorS,errorIndex] = sort(error);  %对距离进行排序  
%         class = floor((errorIndex(1)-1)/perClassTrainLen)+1;%将图像分到距离最小的类别中去,预测的类别  
%           
%         oriclass =  floor((x-1)/perClassTestLen)+1 ; %实际的类别  
%         if(class == oriclass)  
%             right = right+1;  
%         end  
%     end  
%       
%     accuracy = right/numTestInstance;  
      
end