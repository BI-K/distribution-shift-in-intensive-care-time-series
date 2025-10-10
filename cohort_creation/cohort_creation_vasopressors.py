import pandas as pd
import json
from psycopg2 import sql
import mimic_waveform_explore_helper as mweh



def get_medication_administration_durations(records, medication_codes):
    durations_mv = []
    durations_cv = []
    for record in records:
        subject_id = int(record.split('-')[0].replace('p', ''))
        hadm_id = mweh.get_hadm_id_from_record(record, cur)

        cur.execute("SELECT itemid, starttime, endtime FROM mimiciii.inputevents_mv WHERE subject_id=%s AND hadm_id=%s", (subject_id, hadm_id,))
        mv_durations = cur.fetchall()
        for item_id, starttime, endtime in mv_durations:
            if item_id in medication_codes:
                durations_mv.append((subject_id, record, item_id, starttime, endtime))

        cur.execute("SELECT itemid, charttime, amount, amountuom, rate, rateuom, stopped, newbottle, originalrate, originalamount FROM mimiciii.inputevents_cv WHERE subject_id=%s AND hadm_id=%s", (subject_id, hadm_id,))
        cv_durations = cur.fetchall()
        for item_id, charttime, amount, amountuom, rate, rateuom, stopped, new_bottle, original_rate, original_amount in cv_durations:
            if item_id in medication_codes:
                durations_cv.append((subject_id, record, item_id, charttime, amount, amountuom, rate, rateuom, stopped, new_bottle, original_rate, original_amount))

    df_durations_mv = pd.DataFrame(durations_mv, columns=['subject_id', 'record_id', 'item_id', 'starttime', 'endtime'])
    df_durations_cv = pd.DataFrame(durations_cv, columns=['subject_id', 'record_id', 'item_id', 'charttime', 'amount', 'amountuom', 'rate', 'rateuom', 'stopped', 'new_bottle', 'original_rate', 'original_amount'])
    return df_durations_mv, df_durations_cv


# df ['subject_id', 'record_id', 'item_id', 'charttime', 'rate', 'stopped']
def calculate_start_end_time_medication_administrations(df):
    # subject_id, record_id, medication_admin_idx, item_id, starttime, endtime
    record_start_endtimes = []

    # sort df
    df = df.sort_values(by=['subject_id', 'record_id', 'item_id', 'charttime'])


    subject_id = ""
    record_id = ""
    item_id = ""
    medication_admin_idx = 0
    need_to_initialize = True
    
    for idx, row in df.iterrows():

        if subject_id != row['subject_id'] or record_id != row['record_id'] or item_id != row['item_id']:
            # if we are not in the same subject, record, and item, we need to initialize
            need_to_initialize = True

        # initialize variables for new medication administration
        if need_to_initialize:
            subject_id = row['subject_id']
            record_id = row['record_id']
            medication_admin_idx += 1
            item_id = row['item_id']
            starttime = row['charttime']
            need_to_initialize = False

        if row['stopped'] == "Stopped":
            endtime = row['charttime']
            if endtime != starttime:
                record_start_endtimes.append((subject_id, record_id, item_id, starttime, endtime))
            need_to_initialize = True




    return pd.DataFrame(record_start_endtimes, columns=['subject_id', 'record_id', 'item_id', 'chartdate_start', 'chartdate_end'])


# def attach record_start_date record_end_date

# transform to start_end
def transform_csv_to_csv_with_start_and_end(df):
    """
    Transforms a CSV file with records into a new CSV file with start and end times.
    :param input_csv_path: Path to the input CSV file.
    :param output_csv_path: Path to the output CSV file.
    """

    # drop all rows where chartdate_start == chartdate_end
    df = df[df["chartdate_start"] != df["chartdate_end"]]

    path_to_numerics_signal_duration = 'data\mimic3wdb-matched_numerics_signals_duration.csv'
    numerics_signal_duration = pd.read_csv(path_to_numerics_signal_duration)
    numerics_signal_duration = numerics_signal_duration[numerics_signal_duration['record_id'].isin(df['record_id'])]
    numerics_signal_duration = numerics_signal_duration.drop_duplicates(subset=['record_id'], keep='first')
    numerics_signal_duration = numerics_signal_duration.drop(columns=['signal', 'sampling_frequency', 'subject'])

    numerics_signal_duration['record_start_time'] = pd.to_datetime((numerics_signal_duration['record_id'].str.replace('n', '', regex=False).str[8:]), format='%Y-%m-%d-%H-%M')
    numerics_signal_duration['record_end_time'] = (
        pd.to_datetime(numerics_signal_duration['record_start_time'], format='%Y-%m-%d-%H-%M')
        + pd.to_timedelta(numerics_signal_duration['duration_seconds'], unit='s')
    )

    merged_df = pd.merge(df, numerics_signal_duration, left_on='record_id', right_on='record_id', how='inner')
    # drop columns
    #merged_df = merged_df.drop(columns=['duration_seconds'])

    merged_df['record_contains_medication'] = (merged_df['chartdate_start'] >= merged_df['record_start_time']) & (merged_df['chartdate_start'] <= merged_df['record_end_time'])
    # ventilation stops during recording
    merged_df['record_contains_medication'] = merged_df['record_contains_medication'] | (merged_df['chartdate_end'] >= merged_df['record_start_time']) & (merged_df['chartdate_end'] <= merged_df['record_end_time'])
    # recording completely contained in ventilation
    merged_df['record_contains_medication'] = merged_df['record_contains_medication'] | (merged_df['record_start_time'] >= merged_df['chartdate_start']) & (merged_df['record_end_time'] <= merged_df['chartdate_end'])

    merged_df = merged_df[merged_df['record_contains_medication']]
    # drop unnecessary columns record_contains_medication
    merged_df = merged_df.drop(columns=['record_contains_medication'])


    for col in ['record_start_time', 'record_end_time']:
        merged_df[col] = pd.to_datetime(merged_df[col], format='%Y-%m-%d-%H-%M-%S')

    for col in ['chartdate_start','chartdate_end', ]:
        merged_df[col] = pd.to_datetime(merged_df[col], format='%Y-%m-%d %H:%M:%S')

    # Calculate duration in seconds
    merged_df['medication_duration_seconds'] = (
        (pd.concat([merged_df['chartdate_end'], merged_df['record_end_time']], axis=1).min(axis=1) -
        pd.concat([merged_df['chartdate_start'], merged_df['record_start_time']], axis=1).max(axis=1))
        .dt.total_seconds()
    )
    merged_df = merged_df[merged_df["medication_duration_seconds"] >= 3900]

    merged_df = merged_df[merged_df["chartdate_start"] != merged_df["chartdate_end"]]

    merged_df["offset_start_seconds"] = (merged_df["chartdate_start"] - merged_df["record_start_time"]
    ).dt.total_seconds().clip(lower=0)


    merged_df["offset_end_seconds"] = (
        merged_df["offset_start_seconds"] + merged_df["medication_duration_seconds"]
    )

    # Clip to max_duration
    merged_df["offset_end_seconds"] = merged_df["offset_end_seconds"].clip(upper=merged_df["duration_seconds"])

    # filter out all entries where "offset_start_seconds" > "duration_seconds"
    merged_df = merged_df[merged_df["offset_start_seconds"] < merged_df["duration_seconds"]]
    merged_df = merged_df[merged_df["offset_end_seconds"] - merged_df["offset_start_seconds"] >= 3900]


    print("Statistics of continuous Vasopressor duration: ")
    print("unique patients:", merged_df['subject_id'].nunique())
    print("unique records:", merged_df['record_id'].nunique())
    total_duration = merged_df['medication_duration_seconds'].sum()
    print("Total medication duration (seconds):", total_duration)
    print("Total medication duration (days):", total_duration / (60 * 60 * 24))

    merged_df = merged_df.drop(columns=["chartdate_start", "chartdate_end", "record_start_time", "record_end_time", "subject_id", "item_id", "medication_duration_seconds", "duration_seconds"])

    return merged_df

###### Preparation ##################################################################################################

# load valid records
valid_records = []
with open('data/hinrichs_dataset/valid_records_hinrichs_base_model.json', 'r') as f:
    valid_records = json.load(f)

valid_subjects = [s['subject'] for s in valid_records]
valid_subjects = list(set(valid_subjects))

conn, cur = mweh.connect_to_local_mimic_iii()

###### Filter for Records with medication during associated stay ###################################################

# based on mail from Jonas 27.08.2025
# 30042,Dobutamine
# 30043,Dopamine
# 30044,Epinephrine <-
# 30047,Levophed <-
# 30051,Vasopressin
# 30119,Epinephrine-k
# 30120,Levophed-k
# 30127,Neosynephrine
# 30128,Neosynephrine-k
vasopressors = [30042,30043,30044,30047,30051,30119,30120,30127,30128]

records_on_vasopressors = mweh.get_records_on_medication_during_stay(
    medication_item_ids=vasopressors,
    records=valid_records,
    cur=cur
)

print("Statistics on valid records with vasopressors:")
mweh.print_statistics_of_waveform_records(records_on_vasopressors, cur)

###### get records without any vasopressors ########################################################################


valid_records_without_vasopressors = [record for record in valid_records if record not in records_on_vasopressors]
print("Statistics on valid records without vasopressors:")
mweh.print_statistics_of_waveform_records(valid_records_without_vasopressors, cur)

#with open('data/hinrichs_dataset/records_without_vasopressors_surgery_during_stay.txt', 'w') as f:
#    for record in valid_records_without_vasopressors:
#        f.write(f"{record["record_id"]}\n")

#mweh.transtlate_txt_to_csv_with_start_and_end('./data/hinrichs_dataset/records_without_vasopressors_surgery_during_stay.txt',
#                                             './data/hinrichs_dataset/records_with_start_endtime/no_vasopressors.csv')


############ get start and end time of continuous medication administration #########################################

records_on_vasopressors_record_ids = [record["record_id"] for record in records_on_vasopressors]
df_durations_mv, df_durations_cv = get_medication_administration_durations(records_on_vasopressors_record_ids, vasopressors)

# only df_durations_cv contains any entries

df_usable_rate_medications = df_durations_cv[(df_durations_cv['rate'].notna() & df_durations_cv['amount'].isna() & (df_durations_cv['rate'] != 0.0000)) | (df_durations_cv['stopped'] == "Stopped")]
df_usable_rate_medications = df_usable_rate_medications.sort_values(by=['record_id', 'charttime'])


medication_start_end_df = calculate_start_end_time_medication_administrations(df_usable_rate_medications)

final_medication_df = transform_csv_to_csv_with_start_and_end(medication_start_end_df)

final_medication_df.to_csv('./data/hinrichs_dataset/records_with_start_endtime/vasopressors.csv', index=False)