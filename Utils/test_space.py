from nptdms import TdmsFile

tdms = TdmsFile('D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\data\\Umaze_1805\\180505_Umaze\\ca273_behavtest2.tdms')
df_tdms = tdms.as_dataframe()

for idx in df_tdms.loc[0].index:
    if 'Stimulis' in idx:
        print(idx)

a= 1