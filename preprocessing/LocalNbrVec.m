function Ldata = LocalNbrVec(InData,radsiz)
Num = length(InData);
Ldata = zeros(Num,2*radsiz+1);
for k=1:Num
    if k < radsiz+1
        linds = 1:2*radsiz+1;
    elseif k > Num - radsiz
        linds = Num-2*radsiz:Num;
    else
        linds = k-radsiz:k+radsiz;
    end
    Ldata(k,:) = InData(linds);
end
end


