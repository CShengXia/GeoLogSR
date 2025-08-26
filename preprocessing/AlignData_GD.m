function AData = AlignData_GD(Ref,Test)
AData = (Test + (mean(Ref) - mean(Test)))+(Test-mean(Test))*std(Ref)/std(Test);
AData = AData/(max(AData)-min(AData))*(max(Ref)-min(Ref));
AData = AData+(mean(Ref) - mean(AData));