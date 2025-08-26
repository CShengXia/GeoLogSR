function [Tdata,TObj]= MultiScaleRFRdataExt_v(Hdata,Ldata,scale,datalens)
lnum = length(Ldata);
Tdata = zeros(lnum-2,2*scale+1);
TObj = zeros(lnum-2,1);
dinds = [];
for k=1:length(datalens)
    if k==1
       dinds = [dinds;1;datalens(1)];
    elseif k==length(datalens)
       dinds = [dinds;sum(datalens(1:(k-1)))+1;sum(datalens(1:(k-1)))+datalens(k)];
    else
       dinds = [dinds;sum(datalens(1:(k-1)))+1;sum(datalens(1:(k-1)))+datalens(k)];
    end
end
for k=1:lnum
    if ~ismember(k,dinds)
        tinds = Ldata(k,1)-scale:Ldata(k,1)+scale;
        Tdata(k,:) = Hdata(tinds,2);
        TObj(k) = Ldata(k,2);
    end
end