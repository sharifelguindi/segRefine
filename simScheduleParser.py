import numpy as np
import pandas as pd
import datetime

def Name_toMRN(text):
    if '_VDCE' in str(text):
        text = str(text)
        MRN = text[-13:-5]
    else:
        MRN = text
    return MRN

def simnote_to_site(text):

    text = text.upper()

    if ('PROSTATE' in text and not 'BRACHY' in text and not 'SBRT' in text and not 'OP' in text and not 'NODES' in text and not 'MODHYPOFX' in text):
        site = 'PROSTATE'

    elif ('PROSTATE' in text and not 'BRACHY' in text and not 'SBRT' in text and not 'OP' in text and 'NODES' in text and not 'MODHYPOFX' in text):
        site = 'PROSTATE WITH NODES'

    elif ('PROSTATE' in text and not 'BRACHY' in text and not 'SBRT' in text and not 'OP' in text and not'NODES' in text and 'MODHYPOFX' in text):
        site = 'PROSTATE MODHYPO'

    elif ('PROSTATE' in text and not 'BRACHY' in text and not 'SBRT' in text and not 'OP' in text and 'NODES' in text and 'MODHYPOFX' in text):
        site = 'PROSTATE MODHYPO WITH NODES'

    elif ('PROSTATE' in text and not 'BRACHY' in text and 'SBRT' in text and not 'OP' in text and not 'NODES' in text and not 'MODHYPOFX' in text):
        site = 'PROSTATE SBRT'

    elif ('PROSTATE' in text and not 'BRACHY' in text and 'SBRT' in text and not 'OP' in text and 'NODES' in text and not 'MODHYPOFX' in text):
        site = 'PROSTATE SBRT WITH NODES'

    elif ('PROSTATE' in text and 'BRACHY' in text and 'SBRT' in text and not 'OP' in text and not 'NODES' in text and not 'MODHYPOFX' in text):
        site = 'PROSTATE SBRT POST-BRACHY'

    elif ('PROSTATE' in text and 'BRACHY' in text and 'SBRT' in text and not 'OP' in text and 'NODES' in text and not 'MODHYPOFX' in text):
        site = 'PROSTATE SBRT POST-BRACHY WITH NODES'

    elif ('PROSTATE' in text and 'BRACHY' in text and not 'SBRT' in text and not 'OP' in text and not 'NODES' in text and not 'MODHYPOFX' in text):
        site = 'PROSTATE POST-BRACHY'

    elif ('PROSTATE' in text and 'BRACHY' in text and not 'SBRT' in text and not 'OP' in text and 'NODES' in text and not 'MODHYPOFX' in text):
        site = 'PROSTATE POST-BRACHY WITH NODES'

    elif ('PROSTATE' in text or 'FOLEY' in text and not 'BRACHY' in text and not 'SBRT' in text and 'OP' in text and not 'NODES' in text and not 'MODHYPOFX' in text):
        site = 'PROSTATE BED'

    elif ('PROSTATE' in text or 'FOLEY' in text and not 'BRACHY' in text and not 'SBRT' in text and 'OP' in text and 'NODES' in text and not 'MODHYPOFX' in text):
        site = 'PROSTATE BED WITH NODES'

    elif ('H&N' in text or 'NECK' in text or 'HEAD' in text or 'PHARYNX' in text or 'LARYNX' in text or 'ORAL' in text or 'NASO' in text or 'ORO' in text or 'NOSE' in text and not 'FEMORAL' in text):
        site = 'HEAD AND NECK'

    elif ('BREAST' in text or 'CHESTWALL' in text):
        site = 'BREAST'
    else:
        site = 'UNDEFINED'

    return site

def simdate_to_datetime(date, time):

    if not isinstance(time, datetime.datetime):
        try:
            time_formatted = datetime.datetime.strptime(time, ' %I:%M %p')
        except:
            time_formatted = time
    else:
        time_formatted = time

    date_formatted = datetime.datetime.strptime(date, '%m/%d/%Y %a')
    try:
        date_time_object = datetime.datetime.combine(date_formatted.date(), time_formatted)
    except:
        date_time_object = datetime.datetime.combine(date_formatted.date(), time_formatted.time())

    return date_time_object

def fix_mrn(mrn, length):

    while len(mrn) < length:
        mrn = '0' + mrn

    return mrn


# rawDataFeed: path to data from HIS scheduling that needs processing
rawDatafeed = 'G:\\Projects\\AutoQC\\rawDataHIS.xlsx'

# dataPath: path to processed contoured data generated from Matlab
dataPath = 'G:\\Projects\\AutoQC\\data.xlsx'

# questPath: path to Feedback Questionaire
questPath = 'H:\\Treatment Planning\\Elguindi\\prostateSegAnalysis\\metrics\\Feedback.xlsx'

data = pd.read_excel(rawDatafeed)
print('Cleaning site names...')
i = 0
for text in data.SIMNOTE:
    site = simnote_to_site(str(text))
    data.SITE[i] = site
    i = i + 1

print('Creating DateTime Objects...')
i = 0
for time in data.SIMTIME:

    if i == 0:
        date_time_prv = simdate_to_datetime(data.SIMDATE[i], time)
        date_time_obj = date_time_prv
    else:
        date_time_curr = simdate_to_datetime(data.SIMDATE[i], time)

        if date_time_curr.strftime('%Y') == '0001' and i > 0:
            date_time_obj = date_time_prv
        else:
            date_time_prv = date_time_curr
            date_time_obj = date_time_curr

    data.DATETIME[i] = date_time_obj
    i = i + 1

print('Cleaning MRNs...')
data.MRN = data.MRN.astype('str')
i = 0
for mrn in data.MRN:
    data.MRN[i] = fix_mrn(mrn, 8)
    i = i + 1

print('Remove UNDEFINED sites....')
data_cleaned = data.loc[~(data == 'UNDEFINED').any(axis=1)]
data_cleaned.reset_index(drop=True, inplace=True)

new_data = pd.read_excel(dataPath)
print('Cleaning newData MRNs...')
new_data.MRN = new_data.MRN.astype('str')
i = 0
for mrn in new_data.MRN:
    new_data.MRN[i] = fix_mrn(mrn, 8)
    i = i + 1

questData = pd.read_excel(questPath)
print('Cleaning feedback MRNs...')
questData.MRN = questData.MRN.astype('str')
i = 0
for mrn in questData.MRN:
    questData.MRN[i] = fix_mrn(mrn, 8)
    i = i + 1

merged_df = pd.merge(data_cleaned, new_data, how='left', on=['MRN'])
merged_df = pd.merge(merged_df, questData, how='left', on=['MRN'])

databasePath =  'G:\\Projects\\AutoQC\\prostateDB.xlsx'

columnsToDrop = ['SIMDATE', 'SIMTIME', 'Task Campus', 'Task Due Date', 'Task Completion Date', 'Task Status', 'Qn Appr?', 'Qn Appr Date']

merged_df.drop(columns=columnsToDrop, inplace=True)
merged_df.rename(columns={'Qn 3 Resp': 'qualityScore', 'Qn 2 Resp': 'timeSpent', 'Qn 3 Resp.1': 'COMMENTS'}, inplace=True)

merged_df.to_excel(databasePath)