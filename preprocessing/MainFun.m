function [OutParams] = MainFun(LogName,options)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Well Logging SuperResolution MainFun using High resolution ImageRes data%
% Requirements:                                                           %
% 1.模型井数据存放到一个文件夹里，0.125米常规曲线放到（每口井）一个csv文件%
%   里，高分辨电阻率曲线放到（每口井）一个csv文件里。                     %
% 2.保证常规曲线深度范围与高分辨曲线深度范围相同(顶底取整,高分辨深度最接近%
%   整数点或正好整数点)                                                   %
% 3.目前测试井文件夹默认只做1口井的测试，可修改对多井某曲线进行超分辨     %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mode = options.mode;
if strcmp(mode,'train')
    %%% 步骤1：读取高分辨数据
    errT = options.errT;
    treenum = options.treenum;
    [~,HDepths,HResLogs] = HResLogsReading;
    [~,CDepths,ConvLogs] = ConvLogsReading(LogName);
    lnum = length(HResLogs);
    lind = 0;
    mlen = 0;
    for k=1:lnum
        tlen = length(HResLogs{k});
        if tlen > mlen
            mlen = tlen;
            lind = k;
        end
    end
    rinds = setdiff(1:lnum,lind);
    %% 极值约束
    for k=1:lnum
        tmedian = median(abs(HResLogs{k}));
        tmedian_l = median(abs(ConvLogs{k}));
        T = 3*tmedian;
        T_l = 3*tmedian_l;
        HResLogs{k} = max(HResLogs{k},-T);
        HResLogs{k} = min(HResLogs{k},T);
        ConvLogs{k} = max(ConvLogs{k},-T_l);
        ConvLogs{k} = min(ConvLogs{k},T_l);
        HResLogs{k} = OutlierRemove(HResLogs{k},3);
        ConvLogs{k} = OutlierRemove(ConvLogs{k},3);
    end
    %% 对齐到最多数据数据井
    for k=1:lnum-1
        HResLogs{rinds(k)} = AlignData_GD(HResLogs{lind},HResLogs{rinds(k)});
        ConvLogs{rinds(k)} = AlignData_GD(ConvLogs{lind},ConvLogs{rinds(k)});
    end 
    OutParams.alignParams = [mean(ConvLogs{lind}) std(ConvLogs{lind}) min(ConvLogs{lind}) max(ConvLogs{lind})];
    for k=1:lnum
        tDepths = HDepths{k};
        tLog = HResLogs{k};
        tData(:,1) = tDepths;
        tData(:,2) = tLog;
        rData = Resampling_v(tData,1/512);   
        xx = min(rData(:,1)):1/8:max(rData(:,1));
        ttData(:,1) = CDepths{k};
        ttData(:,2) = ConvLogs{k};
        ConvLogs{k} = Resampling_v1(ttData,xx);
        HResLogs512{k} = rData;
        clear tData rData tLog tDepths ttData
    end                   
    HHResLogs512 = [];
    for k=1:lnum
        HHResLogs512 = [HHResLogs512;HResLogs512{k}];
    end
    clear HResLogs;
    HHResLogs512(:,1) = 1:length(HHResLogs512);
    mn = min(HHResLogs512(:,2));
    mx = max(HHResLogs512(:,2));
    HHResLogs512(:,2) = (HHResLogs512(:,2) - mn)/(mx-mn);
    %设计低分辨曲线虚拟序号坐标    
    for k=1:lnum
        ConvLogs{k}(:,2) = Normalize_MinMax(ConvLogs{k}(:,2));       
    end
    LResLogs = [];
    LResInds = [];
    Sdepth = HDepths{1}(1);
    for k=1:lnum
        LResLogs = [LResLogs;ConvLogs{k}(:,2)];
        if k == 1
            LResInds = 1:64:(length(ConvLogs{k})-1)*64+1;
            EndInds(1) = length(LResLogs);
        else
            LResInds = [LResInds max(LResInds)+1:64:max(LResInds)+(length(ConvLogs{k})-1)*64+1];
            EndInds(k) = length(LResInds)- sum(EndInds(1:k-1));       
        end
    end
    LResData(:,1) = single(LResInds);    
    LResData(:,2) = single(LResLogs);
    %   输出endlnds的值      以便于同合计 每个井的数据
    disp(EndInds);
    filepath1 = 'D:\raw_data\GR\HHResLogs512.csv';
    writematrix(HHResLogs512, filepath1);
    filepath2 = 'D:\raw_data\GR\LResData.csv';
    writematrix(LResData, filepath2);
    
    
    
    clear LResLogs LResInds;
    [LHResData512,RFMtrees_L2H] = MultiScaleSuperResolution_v_rf(HHResLogs512,LResData,EndInds,treenum,errT);
    T = single(LHResData512(:,2));
    for k=1:10
        T = medfilt1(T,3);
    end
     
    % 指定文件路径           输出LH的高分辨参考数据
    outputPath = 'D:\raw_data\GR\LHResData512.csv';

    % 将数据写入CSV文件
    writematrix(LHResData512, outputPath);
    
elseif strcmp(mode,'test')
   
end