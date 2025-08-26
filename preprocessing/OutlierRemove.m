function OutLogs = OutlierRemove(InLog,nsigma)
realinds = find(~isnan(InLog));
RealData = InLog(realinds);
[mu, sigma] = normfit(RealData);
thresh1 = mu - nsigma*sigma;
thresh2 = mu + nsigma*sigma;
inds1 = find(RealData < thresh1);
inds2 = find(RealData > thresh2);
inds = union(inds1,inds2);
rinds = setdiff(1:length(RealData),inds);
for k=1:length(inds)
    tind = inds(k);
    dists = abs(rinds - tind);
    [~,sinds] = sort(dists);
    RealData(tind) = mean(RealData(rinds(sinds)));
end
OutLogs = InLog;
OutLogs(realinds) = RealData;


