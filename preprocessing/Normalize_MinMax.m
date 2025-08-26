function Normdata = Normalize_MinMax(InData)
Normdata = (InData - min(InData))/(max(InData) - min(InData));