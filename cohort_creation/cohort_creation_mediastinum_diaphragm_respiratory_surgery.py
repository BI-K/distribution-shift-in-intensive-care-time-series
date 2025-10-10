import pandas as pd
import json
from psycopg2 import sql
import mimic_waveform_explore_helper as mweh

# icd9 codes
# 30-34 Operations On The Respiratory System 
# 34 Operations On Chest Wall, Pleura, Mediastinum, And Diaphragm contained in there
def get_subjects_with_respiratory_surgery(subjects, conn):

    # transform the subject names
    subjects = [str(int(subject.split('/')[1].replace('p',''))) for subject in subjects]
    
    icd9_codes_per_subject = {}
    for subject in subjects:
        cur = conn.cursor()
        query = sql.SQL("SELECT hadm_id, icd9_code FROM mimiciii.procedures_icd WHERE subject_id = %s")
        cur.execute(query, (subject,))
        result = cur.fetchall()
        stay_ids = list(set([res[0] for res in result]))
        info = {stay_id: [] for stay_id in stay_ids}
        for res in result:
            info[res[0]].append(res[1])
        icd9_codes_per_subject[subject] = info

    relevant_subject_stay_ids = []
    for subject, stay_info in icd9_codes_per_subject.items():
        for stay_id, codes in stay_info.items():
            for code in codes:
                if code[0]=='3' and code[1] in ['0','1','2','3','4']:
                    # transform subject_id
                    subject_str = f"p{int(subject):06d}"[:3] + f"/p{int(subject):06d}/"
                    new_entry = {"subject": subject_str, "hadm_id": stay_id}
                    if new_entry not in relevant_subject_stay_ids:
                        relevant_subject_stay_ids.append({"subject": subject_str, "hadm_id": stay_id})

    return relevant_subject_stay_ids


###### Preparation ######################################

# load valid records
valid_records = []
with open('data/hinrichs_dataset/valid_records_hinrichs_base_model.json', 'r') as f:
    valid_records = json.load(f)

valid_subjects = [s['subject'] for s in valid_records]
valid_subjects = list(set(valid_subjects))

conn, cur = mweh.connect_to_local_mimic_iii()


###### selection via icd9 codes ########################

stays_with_respiratory_surgery_icd9 = get_subjects_with_respiratory_surgery(valid_subjects, conn)
records_with_respiratory_surgery_icd9 = mweh.subject_id_and_hadm_id_to_record_id(stays_with_respiratory_surgery_icd9, valid_records, cur)
print("Statistics on records selected via ICD9 codes 30 - 34")
mweh.print_statistics_of_waveform_records(records_with_respiratory_surgery_icd9, cur)



###### selection via cpt_codes #########################

cpt_codes = []
# Surgical Procedures on the Respiratory System
for i in range(30000,33000):
    cpt_codes.append(str(i))
# Surgical Procedures on the Mediastinum and Diaphragm
for i in range(39000,39600):
    cpt_codes.append(str(i))

records_with_respiratory_surgery_cpt = mweh.get_records_with_procedure_during_stay(valid_records, cpt_codes, cur)
print("Statistics on records selected via CPT codes 30000 - 33000 and 39000 - 39600")
mweh.print_statistics_of_waveform_records(records_with_respiratory_surgery_cpt, cur)


###### get records with absolutetly no respiratory surgery ##################

no_respiratory_surgery_records = [record for record in valid_records if record not in records_with_respiratory_surgery_icd9 and record not in records_with_respiratory_surgery_cpt]
print("Statistics for records with absolutely no respiratory surgery")
mweh.print_statistics_of_waveform_records(no_respiratory_surgery_records, cur)

with open('data/hinrichs_dataset/records_absolutely_without_respiratory_surgery_during_stay.txt', 'w') as f:
    for record in no_respiratory_surgery_records:
        f.write(f"{record["record_id"]}\n")

mweh.transtlate_txt_to_csv_with_start_and_end('./data/hinrichs_dataset/records_absolutely_without_respiratory_surgery_during_stay.txt',
                                             './data/hinrichs_dataset/records_with_start_endtime/no_respiratory_surgery.csv')


###### get records with only one icustays and respiratory surgery ##############

all_records_respiratory_surgery= records_with_respiratory_surgery_cpt + records_with_respiratory_surgery_icd9

icu_stay_counts = mweh.check_number_of_icu_stays(all_records_respiratory_surgery, cur)
icu_stay_counts_df = pd.DataFrame(icu_stay_counts.items(), columns=['record_id', 'icu_stay_count'])
icu_stay_counts_df = icu_stay_counts_df[icu_stay_counts_df['icu_stay_count'] == 1]


all_records_respiratory_surgery = []
with open('data/hinrichs_dataset/records_with_respiratory_surgery_cpt_and_icd9.txt', 'w') as f:
    for record_id in icu_stay_counts_df['record_id']:
        f.write(f"{record_id}\n")
        all_records_respiratory_surgery.append({
            "record_id": record_id,
            "subject": record_id.split('-')[0].replace("p", "")
        })


print("Statistics for records with respiratory surgery based on CPT and ICD 9 Codes")
mweh.print_statistics_of_waveform_records(all_records_respiratory_surgery, cur)

mweh.transtlate_txt_to_csv_with_start_and_end('./data/hinrichs_dataset/records_with_respiratory_surgery_cpt_and_icd9.txt', 
                                             './data/hinrichs_dataset/records_with_start_endtime/respiratory_surgery.csv')

