function LHdata = SuperResolutionFromElecRes_v_rf(HRes,RFMModels)
num = length(HRes);
LHdata = zeros(num,2);
LHdata(:,1) = HRes(:,1);
Patchs5 = zeros(5,num);
Patchs7 = zeros(7,num);
Patchs9 = zeros(9,num);
Patchs11 = zeros(11,num);
Patchs13 = zeros(13,num);
Patchs15 = zeros(15,num);
for k=1:num
    if k < 3
        linds5 = 1:5;
        linds7 = 1:7;
        linds9 = 1:9;
        linds11 = 1:11;
        linds13 = 1:13;
        linds15 = 1:15;
    elseif k < 4
        linds5 = k-2:k+2;
        linds7 = 1:7;
        linds9 = 1:9;
        linds11 = 1:11;
        linds13 = 1:13;
        linds15 = 1:15;
    elseif k < 5
        linds5 = k-2:k+2;
        linds7 = k-3:k+3;
        linds9 = 1:9;
        linds11 = 1:11;
        linds13 = 1:13;
        linds15 = 1:15; 
    elseif k < 6
        linds5 = k-2:k+2;
        linds7 = k-3:k+3;
        linds9 = k-4:k+4;
        linds11 = 1:11;
        linds13 = 1:13;
        linds15 = 1:15;   
    elseif k < 7
        linds5 = k-2:k+2;
        linds7 = k-3:k+3;
        linds9 = k-4:k+4;
        linds11 = k-5:k+5;
        linds13 = 1:13;
        linds15 = 1:15;
    elseif k < 8
        linds5 = k-2:k+2;
        linds7 = k-3:k+3;
        linds9 = k-4:k+4;
        linds11 = k-5:k+5;
        linds13 = k-6:k+6;
        linds15 = 1:15;
    elseif k < 9
        linds5 = k-2:k+2;
        linds7 = k-3:k+3;
        linds9 = k-4:k+4;
        linds11 = k-5:k+5;
        linds13 = k-6:k+6;
        linds15 = k-7:k+7;
    elseif k == num - 7
        linds15 = num-14:num;
        linds13 = k-6:k+6;
        linds11 = k-5:k+5;
        linds9 = k-4:k+4;
        linds7 = k-3:k+3;
        linds5 = k-2:k+2;        
    elseif k == num - 6
        linds15 = num-14:num;
        linds13 = num-12:num;
        linds11 = k-5:k+5;
        linds9 = k-4:k+4;
        linds7 = k-3:k+3;
        linds5 = k-2:k+2;   
    elseif k == num - 5
        linds15 = num-14:num;
        linds13 = num-12:num;
        linds11 = num-10:num;
        linds9 = k-4:k+4;
        linds7 = k-3:k+3;
        linds5 = k-2:k+2;   
    elseif k == num - 4
        linds15 = num-14:num;
        linds13 = num-12:num;
        linds11 = num-10:num;
        linds9 = num-8:num;
        linds7 = k-3:k+3;
        linds5 = k-2:k+2;  
    elseif k == num - 3
        linds15 = num-14:num;
        linds13 = num-12:num;
        linds11 = num-10:num;
        linds9 = num-8:num;
        linds7 = num-6:num;
        linds5 = k-2:k+2;  
    elseif k >= num - 2
        linds15 = num-14:num;
        linds13 = num-12:num;
        linds11 = num-10:num;
        linds9 = num-8:num;
        linds7 = num-6:num;
        linds5 = num-4:num; 
    else
        linds5 = k-2:k+2;
        linds7 = k-3:k+3;
        linds9 = k-4:k+4;
        linds11 = k-5:k+5;
        linds13 = k-6:k+6;
        linds15 = k-7:k+7;
    end
    Patchs5(:,k) = HRes(linds5,2);
    Patchs7(:,k) = HRes(linds7,2);
    Patchs9(:,k) = HRes(linds9,2);
    Patchs11(:,k) = HRes(linds11,2);
    Patchs13(:,k) = HRes(linds13,2);
    Patchs15(:,k) = HRes(linds15,2);
end
pre5 = predict(RFMModels{1},Patchs5');
pre7 = predict(RFMModels{2},Patchs7');
pre9 = predict(RFMModels{3},Patchs9');
pre11 = predict(RFMModels{4},Patchs11');
pre13 = predict(RFMModels{5},Patchs13');
pre15 = predict(RFMModels{6},Patchs15');
Pres = [pre5 pre7 pre9 pre11 pre13 pre15];
Pre = predict(RFMModels{7},Pres);
Pre = medfilt1(Pre,3);
TreeNum = length(RFMModels);
if TreeNum > 7
    for k=8:TreeNum
        Tdata = LocalNbrVec(Pre,2);
        Pre = predict(RFMModels{k},Tdata);
        Pre = medfilt1(Pre,3);
    end
end
LHdata(:,2) = Pre;

