# Run this script to generate `metainfo.csv`

import pandas as pd

fn = ['1001_N0D1',
 '1003_I1D1',
 '1005_I3D1',
 '1007_O1D1',
 '1009_O1R2',
 '1011_O3R2',
 '1013_B1D2',
 '1015_B3D2',
 '3002_I1D1',
 '3004_O1D1',
 '3006_B1D1',
 '3008_N0D2',
 '3010_I3D2',
 '3012_O3D2',
 '3014_B3D2',
 '1002_N0D2',
 '1004_I1D2',
 '1006_I3D2',
 '1008_O1L1',
 '1010_O3L1',
 '1012_B1D1',
 '1014_B3D1',
 '3001_N0D1',
 '3003_I3D1',
 '3005_O3D1',
 '3007_B3D1',
 '3009_I1D2',
 '3011_O1D2',
 '3013_B1D2']

_CM = {
    'N': 'None',
    'I': 'InnerRace',
    'O': 'OuterRace',
    'B': 'Ball'
}

dd = []
for f in fn:
    dd.append(_CM[f[5]])

# _DM = {
#     'D1': 'Defect 1',
#     'D2': 'Defect 2',
#     'L1': 'Defect 1',
#     'R2': 'Defect 2',
# }

_DM = {
    'D1': 1,
    'D2': 2,
    'L1': 1,
    'R2': 2,
}

de = []
for f in fn:
    de.append(_DM[f[7:]])

df = pd.DataFrame({'ID': fn, 'FaultComponent': dd, 'FaultExtend': de})

df.to_csv('metainfo.csv', index=False)