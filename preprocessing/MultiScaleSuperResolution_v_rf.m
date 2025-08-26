function [SuperSig,RFRTrees] = MultiScaleSuperResolution_v_rf(HRdata,LRdata,datalens,treenum,errT)
[Tdata5,TObj5]= MultiScaleRFRdataExt_v(HRdata,LRdata,2,datalens);
[Tdata7,TObj7]= MultiScaleRFRdataExt_v(HRdata,LRdata,3,datalens);
[Tdata9,TObj9]= MultiScaleRFRdataExt_v(HRdata,LRdata,4,datalens);
[Tdata11,TObj11]= MultiScaleRFRdataExt_v(HRdata,LRdata,5,datalens);
[Tdata13,TObj13]= MultiScaleRFRdataExt_v(HRdata,LRdata,6,datalens);
[Tdata15,TObj15]= MultiScaleRFRdataExt_v(HRdata,LRdata,7,datalens);
RFModel5 = TreeBagger(treenum,Tdata5,TObj5,'method','regression');
RFModel7 = TreeBagger(treenum,Tdata7,TObj7,'method','regression');
RFModel9 = TreeBagger(treenum,Tdata9,TObj9,'method','regression');
RFModel11 = TreeBagger(treenum,Tdata11,TObj11,'method','regression');
RFModel13 = TreeBagger(treenum,Tdata13,TObj13,'method','regression');
RFModel15 = TreeBagger(treenum,Tdata15,TObj15,'method','regression');
RFRTrees{1} = RFModel5;
RFRTrees{2} = RFModel7;
RFRTrees{3} = RFModel9;
RFRTrees{4} = RFModel11;
RFRTrees{5} = RFModel13;
RFRTrees{6} = RFModel15;
pre5  = predict(RFModel5,Tdata5);
pre7  = predict(RFModel7,Tdata7);
pre9  = predict(RFModel9,Tdata9);
pre11  = predict(RFModel11,Tdata11);
pre13  = predict(RFModel13,Tdata13);
pre15  = predict(RFModel15,Tdata15);
Pres = [pre5 pre7 pre9 pre11 pre13 pre15];
Trees = TreeBagger(treenum,Pres,TObj15,'method','regression');
RFRTrees{7} = Trees;
Pre = predict(Trees,Pres);
err = sum(abs(Pre - TObj5))/length(TObj5);
done = 0;
% Pre = medfilt1(Pre,3);
if err < errT
    done = 1;
end
n = 0;
while ~done
    n = n+1;
    Tdata = LocalNbrVec(Pre,2);
    Trees = TreeBagger(treenum,Tdata,TObj15,'method','regression');
    RFRTrees{7+n} = Trees;
    Pre = predict(Trees,Tdata);
%     Pre = medfilt1(Pre,3);
    err = sum(abs(Pre - TObj5))/length(TObj5);
    if err < errT || n > 4
        done = 1;
    end
end
SuperSig = SuperResolutionFromElecRes_v_rf(HRdata,RFRTrees);
SuperSig(:,2) = OutlierRemove(SuperSig(:,2),3);



end



