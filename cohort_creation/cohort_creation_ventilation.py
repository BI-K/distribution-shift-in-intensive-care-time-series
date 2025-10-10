import pandas as pd
import json
from psycopg2 import sql
import datetime
import mimic_waveform_explore_helper as mweh



def get_start_of_subsequent_ventilation_days(records):
    data = []
    for cpt_code in ['94003']:
        for entry in records:
            subject_id = str(int(entry["subject"].split('/')[1].replace('p', '')))
            record = entry["record_id"]
            hadm_id = mweh.get_hadm_id_from_record(record, cur)

            cur.execute("SELECT chartdate, cpt_cd FROM mimiciii.cptevents WHERE subject_id=%s AND hadm_id=%s AND cpt_cd=%s", (subject_id, hadm_id, cpt_code,))
            cpt_events = cur.fetchall()
            for chartdate, cpt_cd in cpt_events:
                data.append((subject_id, hadm_id, record, cpt_cd, chartdate))

    return pd.DataFrame(data, columns=['subject_id', 'hadm_id', 'record', 'cpt_cd', 'chartdate'])



###### Preparation ######################################

# load valid records
valid_records = []
with open('data/hinrichs_dataset/valid_records_hinrichs_base_model.json', 'r') as f:
    valid_records = json.load(f)

valid_subjects = [s['subject'] for s in valid_records]
valid_subjects = list(set(valid_subjects))

conn, cur = mweh.connect_to_local_mimic_iii()

##### get records with ventilation cpt codes ############

records_with_procedure_stay = mweh.get_records_with_procedure_during_stay(
    records=valid_records,
    cpt_codes=["94002", "94003"],
    cur=cur
)

##### get records without any ventilation during stay ####################

valid_records_without_ventilation = [record for record in valid_records if record not in records_with_procedure_stay]
mweh.print_statistics_of_waveform_records(valid_records_without_ventilation, cur)

with open('data/hinrichs_dataset/records_without_mechanical_ventilation_during_stay.txt', 'w') as f:
    for record in valid_records_without_ventilation:
        f.write(f"{record["record_id"]}\n")

mweh.transtlate_txt_to_csv_with_start_and_end('./data/hinrichs_dataset/records_without_mechanical_ventilation_during_stay.txt', 
                                             './data/hinrichs_dataset/records_with_start_endtime/no_ventilation_records.csv')


####### get start and end times of ventilation consecutive days ##########

ventilation_df = get_start_of_subsequent_ventilation_days(records_with_procedure_stay)
# sort by subject_id, record_id and chartdate + make one df for first and one for last day of consecutive ventilation
ventilation_df = ventilation_df.sort_values(by=['subject_id', 'record', 'chartdate'])
ventilation_df_start = ventilation_df[~ventilation_df.duplicated(subset=['record'], keep='first')]
ventilation_df_end = ventilation_df[~ventilation_df.duplicated(subset=['record'], keep='last')]

# merge the start and end dataframes 
ventilation_df = pd.merge(ventilation_df_start, ventilation_df_end, on=['subject_id', 'hadm_id', 'record'], suffixes=('_start', '_end'))
ventilation_df = ventilation_df.drop(columns=['cpt_cd_start', 'cpt_cd_end'])

# read durations of recordings
path_to_numerics_signal_duration = 'data\mimic3wdb-matched_numerics_signals_duration.csv'
numerics_signal_duration = pd.read_csv(path_to_numerics_signal_duration)

# filter out all columns where record_id is not in ventilation_df + remove duplicated records - so keep the start + drop unneded columns
numerics_signal_duration = numerics_signal_duration[numerics_signal_duration['record_id'].isin(ventilation_df['record'])]
numerics_signal_duration = numerics_signal_duration.drop_duplicates(subset=['record_id'], keep='first')
numerics_signal_duration = numerics_signal_duration.drop(columns=['signal', 'sampling_frequency', 'subject'])

numerics_signal_duration['record_start_time'] = (numerics_signal_duration['record_id'].str.replace('n', '', regex=False).str[8:])
numerics_signal_duration['record_end_time'] = (
    pd.to_datetime(numerics_signal_duration['record_start_time'], format='%Y-%m-%d-%H-%M')
    + pd.to_timedelta(numerics_signal_duration['duration_seconds'], unit='s')
).dt.strftime('%Y-%m-%d-%H-%M-%S')


# merge numerics_signal_duration and ventilation_df on record_id
merged_df = pd.merge(ventilation_df, numerics_signal_duration, left_on='record', right_on='record_id', how='inner')
#merged_df = merged_df.drop(columns=['record_id', 'duration_seconds'])
# add "-00-00-00" to chartdate_start and chartdate_end for unified format
merged_df['chartdate_start'] = merged_df['chartdate_start'].astype(str) + '-00-00-00'
merged_df['chartdate_end'] = merged_df['chartdate_end'].astype(str) + '-00-00-00'
merged_df['record_start_time'] = merged_df['record_start_time'].astype(str) + '-00'


# ventilation starts during recording
merged_df['record_contains_ventilation'] = (merged_df['chartdate_start'] >= merged_df['record_start_time']) & (merged_df['chartdate_start'] <= merged_df['record_end_time'])
# ventilation stops during recording
merged_df['record_contains_ventilation'] = merged_df['record_contains_ventilation'] | (merged_df['chartdate_end'] >= merged_df['record_start_time']) & (merged_df['chartdate_end'] <= merged_df['record_end_time'])
# recording completely contained in ventilation
merged_df['record_contains_ventilation'] = merged_df['record_contains_ventilation'] | (merged_df['record_start_time'] >= merged_df['chartdate_start']) & (merged_df['record_end_time'] <= merged_df['chartdate_end'])

# drop all rows where record_contains_ventilation is False
merged_df = merged_df[merged_df['record_contains_ventilation']]
# drop unnecessary columns record_contains_ventilation
merged_df = merged_df.drop(columns=['record_contains_ventilation'])

# merged_df['ventilation_duration_seconds'] = (min(merged_df['chartdate_end'], merged_df['record_end_time']) - max(merged_df['chartdate_start'], merged_df['record_start_time'])).dt.total_seconds()
for col in ['chartdate_start', 'record_start_time', 'chartdate_end', 'record_end_time']:
    merged_df[col] = pd.to_datetime(merged_df[col], format='%Y-%m-%d-%H-%M-%S')

# Calculate duration in seconds
merged_df['ventilation_duration_seconds'] = (
    (pd.concat([merged_df['chartdate_end'], merged_df['record_end_time']], axis=1).min(axis=1) -
     pd.concat([merged_df['chartdate_start'], merged_df['record_start_time']], axis=1).max(axis=1))
    .dt.total_seconds()
)

# drop columns where ventilation_duration_seconds == 0
merged_df = merged_df[merged_df['ventilation_duration_seconds'] > 0]

# drop all rows where chartdate_start == chartdate_end
merged_df = merged_df[merged_df["chartdate_start"] != merged_df["chartdate_end"]]

merged_df["offset_start_seconds"] = (
    pd.to_datetime(merged_df["chartdate_start"], format='%Y-%m-%d-%H-%M-%S') 
    - pd.to_datetime(merged_df["record_start_time"], format='%Y-%m-%d-%H-%M-%S')
).dt.total_seconds().clip(lower=0)

merged_df = merged_df[merged_df["offset_start_seconds"] < merged_df["duration_seconds"]]

# Calculate the element-wise maximum between chartdate_start and record_start_time
start_max = pd.concat([
    pd.to_datetime(merged_df["chartdate_start"], format='%Y-%m-%d-%H-%M-%S'),
    pd.to_datetime(merged_df["record_start_time"], format='%Y-%m-%d-%H-%M-%S')
], axis=1).max(axis=1)

merged_df["offset_end_seconds"] = (
    pd.to_datetime(merged_df["chartdate_end"], format='%Y-%m-%d-%H-%M-%S') 
    - start_max
).dt.total_seconds().clip(upper=(
    pd.to_datetime(merged_df["record_end_time"], format='%Y-%m-%d-%H-%M-%S') 
    - pd.to_datetime(merged_df["record_start_time"], format='%Y-%m-%d-%H-%M-%S')
).dt.total_seconds())


# drop rows where ventilation_duration_seconds < 65 minutes
merged_df = merged_df[merged_df['ventilation_duration_seconds'] >= 3900]

print("Statistics of Ventilation duration: ")
print("unique patients:", merged_df['subject_id'].nunique())
print("unique records:", merged_df['record'].nunique())
print("unique hadm_ids:", merged_df['hadm_id'].nunique())
total_duration = merged_df['ventilation_duration_seconds'].sum()
print("Total ventilation duration (seconds):", total_duration)
print("Total ventilation duration (days):", total_duration / (60 * 60 * 24))


# save to csv
merged_df = merged_df.drop(columns=["record_id","ventilation_duration_seconds", "chartdate_start", "chartdate_end", "record_start_time", "record_end_time", "subject_id", "hadm_id", "duration_seconds"])
merged_df.to_csv('data/hinrichs_dataset/records_with_start_endtime/ventilation_records.csv', index=False)