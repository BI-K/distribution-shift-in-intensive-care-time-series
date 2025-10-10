import pandas as pd
import json
from psycopg2 import sql
import mimic_waveform_explore_helper as mweh

# icd9 codes
# 35-39 Operations On The Cardiovascular System

###### Preparation ######################################

# load valid records
valid_records = []
with open('data/hinrichs_dataset/valid_records_hinrichs_base_model.json', 'r') as f:
    valid_records = json.load(f)

valid_subjects = [s['subject'] for s in valid_records]
valid_subjects = list(set(valid_subjects))

conn, cur = mweh.connect_to_local_mimic_iii()


###### selection via icd9 codes ########################

stays_with_cardiac_surgery_icd9 = mweh.get_subjects_with_cardiac_surgery(valid_subjects, conn)
records_with_cardiac_surgery_icd9 = mweh.subject_id_and_hadm_id_to_record_id(stays_with_cardiac_surgery_icd9, valid_records, cur)
print("Statistics on records selected via ICD9 codes 35 - 39")
mweh.print_statistics_of_waveform_records(records_with_cardiac_surgery_icd9, cur)



###### selection via cpt_codes #########################

cpt_codes = []
# 33016-37799 Surgical Procedures on the Cardiovascular System
for i in range(33016,37800):
    cpt_codes.append(str(i))

records_with_cardiac_surgery_cpt = mweh.get_records_with_procedure_during_stay(valid_records, cpt_codes, cur)
print("Statistics on records selected via CPT codes 33016 - 37800")
mweh.print_statistics_of_waveform_records(records_with_cardiac_surgery_cpt, cur)


###### get records with absolutetly no cardiac surgery ##################

no_cardiac_surgery_records = [record for record in valid_records if record not in records_with_cardiac_surgery_icd9 and record not in records_with_cardiac_surgery_cpt]
print("Statistics for records with absolutely no cardiac surgery")
mweh.print_statistics_of_waveform_records(no_cardiac_surgery_records, cur)

with open('data/hinrichs_dataset/records_absolutely_without_cardiac_surgery_during_stay.txt', 'w') as f:
    for record in no_cardiac_surgery_records:
        f.write(f"{record["record_id"]}\n")

mweh.transtlate_txt_to_csv_with_start_and_end('./data/hinrichs_dataset/records_absolutely_without_cardiac_surgery_during_stay.txt',
                                             './data/hinrichs_dataset/records_with_start_endtime/no_cardiac_surgery.csv')


###### get records with only one icustays and cardiac surgery ##############

all_records_cardiac_surgery= records_with_cardiac_surgery_cpt + records_with_cardiac_surgery_icd9

icu_stay_counts = mweh.check_number_of_icu_stays(all_records_cardiac_surgery, cur)
icu_stay_counts_df = pd.DataFrame(icu_stay_counts.items(), columns=['record_id', 'icu_stay_count'])
icu_stay_counts_df = icu_stay_counts_df[icu_stay_counts_df['icu_stay_count'] == 1]


all_records_cardiac_surgery = []
with open('data/hinrichs_dataset/records_with_cardiac_surgery_cpt_and_icd9.txt', 'w') as f:
    for record_id in icu_stay_counts_df['record_id']:
        f.write(f"{record_id}\n")
        all_records_cardiac_surgery.append({
            "record_id": record_id,
            "subject": record_id.split('-')[0].replace("p", "")
        })


print("Statistics for records with cardiac surgery based on CPT and ICD 9 Codes")
mweh.print_statistics_of_waveform_records(all_records_cardiac_surgery, cur)

mweh.transtlate_txt_to_csv_with_start_and_end('./data/hinrichs_dataset/records_with_cardiac_surgery_cpt_and_icd9.txt', 
                                             './data/hinrichs_dataset/records_with_start_endtime/cardiac_surgery.csv')

