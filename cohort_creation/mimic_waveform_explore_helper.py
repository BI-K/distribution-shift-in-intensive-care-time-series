import wfdb
import psycopg2
from psycopg2 import sql
import pandas as pd
import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
import time

################################################################################
################ Absolute Helpers ##############################################
################################################################################

def record_id_to_datetime(record_id):
    """
    Convert a record_id to a datetime object.
    """
    record_day = record_id.split('-')[3]
    record_month = record_id.split('-')[2]
    record_year = record_id.split('-')[1]
    record_hour = record_id.split('-')[4]
    record_minute = record_id.split('-')[5].replace('n', '')
    
    date_time = datetime.datetime(int(record_year), int(record_month), int(record_day), 
                                  int(record_hour), int(record_minute))
    
    return date_time




#################################################################################
########## Local MIMIC-III Database Connection and Queries ######################
#################################################################################

def connect_to_local_mimic_iii():
    conn = psycopg2.connect(
        dbname='mimic_iii',
        user='postgres',
        password='1207',
        host='localhost',
        port='5432'
    )

    # connect to the database
    cur = conn.cursor()
    return conn, cur

def test_connection_to_local_mimic_iii():
    try:
        conn, cur = connect_to_local_mimic_iii()
        # perform a simple query to test the connection
        cur.execute("SELECT 1;")
        result = cur.fetchone()
        if result[0] == 1:

            print("Connection to local MIMIC-III database successful.")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error connecting to local MIMIC-III database: {e}")

def get_icd9_codes_mimic_iii():
    conn, cur = connect_to_local_mimic_iii()
    query = sql.SQL("SELECT icd9_code FROM mimic_iii.d_icd_diagnoses")
    cur.execute(query)
    icd9_codes = cur.fetchall()
    cur.close()
    conn.close()
    return icd9_codes


def get_subjects_with_cardiac_surgery(subjects, conn):

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
                if code[0]=='3' and code[1] in ['5', '6', '7', '8', '9']:
                    # transform subject_id
                    subject_str = f"p{int(subject):06d}"[:3] + f"/p{int(subject):06d}/"
                    new_entry = {"subject": subject_str, "hadm_id": stay_id}
                    if new_entry not in relevant_subject_stay_ids:
                        relevant_subject_stay_ids.append({"subject": subject_str, "hadm_id": stay_id})

    return relevant_subject_stay_ids




def attach_start_and_end_time_to_records_under_procedure(procedure_item_ids, records, cur):
    """
    Attach start and end times to records where the patient underwent a specific procedure during their stay.
    
    Args:
        procedure_item_ids (list): List of item IDs for the procedures.
        records (list): List of records to check.
        cur: Database cursor.
        
    Returns:
        list: Records with start and end times for the specified procedure.
    """
    valid_records = []
    
    for record in records:
        subject_id = int(record['record_id'].split('-')[0].replace('p', ''))
        hadm_id = get_hadm_id_from_record(record['record_id'], cur)
        
        cur.execute("SELECT starttime, endtime, itemid FROM mimiciii.procedureevents_mv WHERE subject_id=%s AND hadm_id=%s", (subject_id, hadm_id))
        items_mv = cur.fetchall()
        
        #cur.execute("SELECT charttime, endtime, itemid FROM mimiciii.procedureevents_cv WHERE subject_id=%s AND hadm_id=%s", (subject_id, hadm_id))
        #items_cv = cur.fetchall()
        
        all_items = items_mv #+ items_cv
        
        for item in all_items:
            if item[2] in procedure_item_ids:
                valid_records.append({
                    'record_id': record['record_id'],
                    'subject': record['subject'],
                    'start_time': item[0],
                    'end_time': item[1],
                    'itemid': item[2]
                })
    
    return valid_records





# TODO check how this can be done in the best way - medication also has an effect after it was adiminstered
def attach_start_and_end_times_to_records_on_medication(medication_item_ids, records, cur):
    """
    Attach start and end times to records where the patient was on a specific medication during their stay.
    
    Args:
        medication_item_ids (list): List of item IDs for the medications.
        records (list): List of records to check.
        cur: Database cursor.
        
    Returns:
        list: Records with start and end times for the specified medication.
    """
    valid_records = []
    
    for record in records:
        subject_id = int(record['record_id'].split('-')[0].replace('p', ''))
        hadm_id = get_hadm_id_from_record(record['record_id'], cur)
        
        cur.execute("SELECT starttime, endtime, itemid FROM mimiciii.inputevents_mv WHERE subject_id=%s AND hadm_id=%s", (subject_id, hadm_id))
        items_mv = cur.fetchall()
        
        #cur.execute("SELECT charttime, endtime, itemid FROM mimiciii.inputevents_cv WHERE subject_id=%s AND hadm_id=%s", (subject_id, hadm_id))
        #items_cv = cur.fetchall()
        
        all_items = items_mv + items_cv
        
        for item in all_items:
            if item[2] in medication_item_ids:
                valid_records.append({
                    'record_id': record['record_id'],
                    'subject': record['subject'],
                    'start_time': item[0],
                    'end_time': item[1],
                    'itemid': item[2]
                })
    
    return valid_records



def get_records_on_medication_during_stay(medication_item_ids, records, cur):
    """
    Get records where the patient was on a specific medication during their stay.
    
    Args:
        medication_item_ids (list): List of item IDs for the medications.
        records (list): List of records to check.
        cur: Database cursor.
        
    Returns:
        list: Records where the patient was on the specified medication.
    """
    valid_records = []
    
    for record in records:
        subject_id = int(record['record_id'].split('-')[0].replace('p', ''))
        hadm_id = get_hadm_id_from_record(record['record_id'], cur)
        
        cur.execute("SELECT itemid FROM mimiciii.inputevents_mv WHERE subject_id=%s AND hadm_id=%s", (subject_id, hadm_id))
        items_mv = cur.fetchall()
        
        cur.execute("SELECT itemid FROM mimiciii.inputevents_cv WHERE subject_id=%s AND hadm_id=%s", (subject_id, hadm_id))
        items_cv = cur.fetchall()
        
        all_items = set([item[0] for item in items_mv + items_cv])
        
        if any(item in all_items for item in medication_item_ids):
            valid_records.append(record)
    
    return valid_records





def get_records_with_procedure_during_stay(records, cpt_codes, cur):
    """
    Get records that have mechanical ventilation during stay.
    :param records: List of records to check.
    :param cur: Database cursor.
    :return: List of records with mechanical ventilation during stay.
    """
    records_with_procedure = []
    for record in records:
        record_id = record['record_id']
        subject_id = int(record_id.split('-')[0].replace('p', ''))
        hadm_id = get_hadm_id_from_record(record_id, cur)

        cur.execute("SELECT cpt_cd, description FROM mimiciii.cptevents WHERE subject_id=%s AND hadm_id=%s", (subject_id,hadm_id,))
        item_ids_mv = cur.fetchall()
        if item_ids_mv:
            if any(cpt_code in item_id[0] for cpt_code in cpt_codes for item_id in item_ids_mv):
                records_with_procedure.append(record)

    return records_with_procedure



def subject_id_and_hadm_id_to_record_id(valid_records_with_characteristic, valid_records, cur, check_also_in_admissions=False):
    """
    Convert subject_id and hadm_id to record_id.
    """
    record_id_date_dict = []
    # request date from mimic iii mached waveform
    # read header
    for entry in valid_records:
        subject = entry['subject']
        record_id = entry['record_id']
        
        date_time = record_id_to_datetime(record_id)

        record_id_date_dict.append({
            'subject': subject,
            'record_id': record_id,
            'start_date_time': date_time#,
        #    'end_date_time': end_time
        })

    # get dict of start and end times for each hadm_id in valid_records_with_characteristic
    hadm_id_start_end_date_dict = []
    for entry in valid_records_with_characteristic:
        subject = entry['subject']
        hadm_id = entry['hadm_id']

        # check in ADMISSIONS
        if check_also_in_admissions:

            cur.execute("SELECT admittime, dischtime FROM mimiciii.admissions WHERE hadm_id=%s", (hadm_id,))
            result = cur.fetchone()
            # if match found in ADMISSIONS
            if result:
                hadm_id_start_end_date_dict.append({
                        'subject': subject,
                        'hadm_id': hadm_id,
                        'intime': result[0] - datetime.timedelta(hours=2) if result[0] else None,
                        'outtime': result[1] + datetime.timedelta(hours=2) if result[1] else None
                })

            else:
        
                cur.execute("SELECT intime, outtime FROM mimiciii.icustays WHERE hadm_id=%s", (hadm_id,))
                results = cur.fetchall()
                # if match found in ICUSTAYS
                if results:
                        for result in results:
                            hadm_id_start_end_date_dict.append({
                                'subject': subject,
                                'hadm_id': hadm_id,
                                'intime': result[0] - datetime.timedelta(hours=2) if result[0] else None,
                                'outtime': result[1] + datetime.timedelta(hours=2) if result[1] else None
                            })
        else:

            cur.execute("SELECT intime, outtime FROM mimiciii.icustays WHERE hadm_id=%s", (hadm_id,))
            results = cur.fetchall()
            # if match found in ICUSTAYS
            if results:
                for result in results:
                    hadm_id_start_end_date_dict.append({
                        'subject': subject,
                        'hadm_id': hadm_id,
                        'intime': result[0] - datetime.timedelta(hours=2) if result[0] else None,
                        'outtime': result[1] + datetime.timedelta(hours=2) if result[1] else None
                    })


    print(len(hadm_id_start_end_date_dict), "hadm_id start and end date entries found.")
    translated_record_with_condition = []
    for e in hadm_id_start_end_date_dict:
        # filter record_id_date_dict for subject
        subject = e['subject']
        releveant_record = [r for r in record_id_date_dict if r['subject'] == subject]
        for r in releveant_record:
            # check if base_date is between admittime_date and dischtime_date
            if (e['intime'] <= r['start_date_time']) and  (r['start_date_time'] <= e['outtime']):
                new_entry = {
                    'subject': subject,
                    'record_id': r['record_id']
                }
                if not new_entry in translated_record_with_condition:
                    translated_record_with_condition.append(new_entry)


    return translated_record_with_condition



def get_master_record_id_from_record_id(record_id, subject, cur):
    # get all records for the record_id
    if record_id.endswith('n'):
        record_date = record_id_to_datetime(record_id)
    else:
        print(subject)
        print(record_id)
        all_records = wfdb.get_record_list(f'mimic3wdb-matched/1.0/{subject}')
        all_records = [r for r in all_records if '_' not in r]
        all_records = [r for r in all_records if not r.endswith('n')]

        for record in all_records:
            wfdb_header = wfdb.rdheader(record, pn_dir=f'mimic3wdb-matched/{subject}', rd_segments=True)
            # field of header
            pass
    return ""


def get_hadm_id_from_record(record_id, cur, check_admissions=False):
    """
    Get hadm_id from record_id.
    """
    record_date = record_id_to_datetime(record_id)

    # get subject_id from record_id
    subject_id = int(record_id.split('-')[0].replace('p', ''))

    if check_admissions:
        # get all hadm_ids and admittime and dischtime from ADMISSIONS
        cur.execute("SELECT hadm_id, admittime, dischtime FROM mimiciii.admissions WHERE subject_id=%s", (subject_id,))
        
    else:
        cur.execute("SELECT hadm_id, intime, outtime FROM mimiciii.icustays WHERE subject_id=%s", (subject_id,))
        
    results = cur.fetchall()
    for result in results:
        hadm_id = result[0]
        admittime = result[1] - datetime.timedelta(hours=2)
        dischtime = result[2]  + datetime.timedelta(hours=2)

        # check if record_date is between admittime and dischtime
        if admittime <= record_date <= dischtime:
            return hadm_id

    return None

def get_subject_demographics(subject_record_dict_list, cur, check_admissions=False):
    """
    Get subject demographics from the MIMIC-III database.
    """
    # get hadm_id
    results_demographics = []
    i = 0
    for entry in subject_record_dict_list:
        subject = entry['subject'].split('/')[1].replace('p', '')
        hadm_id = get_hadm_id_from_record(entry['record_id'], cur, check_admissions=check_admissions)
        if hadm_id is None:
            print(f"Warning: No hadm_id found for record_id {entry['record_id']}. Skipping this entry.")
            i += 1
            continue
        
        # calculate age in years
        cur.execute("SELECT dob, gender FROM mimiciii.patients WHERE subject_id=%s", (subject,))
        result = cur.fetchone()
        if result is None:
            print(f"Warning: No patient found for subject_id {subject}. Skipping this entry.")
            continue
        dob = result[0]
        gender = result[1]

        cur.execute("SELECT admittime, ethnicity FROM mimiciii.admissions WHERE hadm_id=%s", (hadm_id,))
        result = cur.fetchone()
        if result is None:
            print(f"Warning: No admission found for hadm_id {hadm_id}. Skipping this entry.")
            continue
        admittime = result[0]
        ethnicity = result[1]

        age_years = (admittime - dob).days // 365

        # get diagnosis and procedures
        diagnoses = get_diagnoses(hadm_id, cur)
        procedures = get_procedures(hadm_id, cur)


        results_demographics.append({
            'subject': subject,
            'record_id': entry['record_id'],
            'hadm_id': hadm_id,
            'age_years': age_years,
            'gender': gender,
            'ethnicity': ethnicity,
            'diagnoses': diagnoses,
            'procedures': procedures
        })
    
    print(f"No hadm_id found for {i} records.")
    # turn results_demographics into a pandas DataFrame
    demographics_df = pd.DataFrame(results_demographics)

    return demographics_df

def get_diagnoses(hadm_id, cur):
    """
    Get diagnoses statistics from the MIMIC-III database.
    """
    # get icd9_code from DIAGNOSES_ICD
    cur.execute("SELECT icd9_code FROM mimiciii.diagnoses_icd WHERE hadm_id=%s", (hadm_id,))
    results = cur.fetchall()
    diagnoses = [result[0] for result in results]
    return diagnoses

def get_procedures(hadm_id, cur):
    """
    Get procedures statistics from the MIMIC-III database.
    """
    # get icd9_code from PROCEDURES_ICD
    cur.execute("SELECT icd9_code FROM mimiciii.procedures_icd WHERE hadm_id=%s", (hadm_id,))
    results = cur.fetchall()
    procedures = [result[0] for result in results]
    return procedures

##################################################################################
########## MIMIC-III Waveform Database Queries ###################################
##################################################################################

database_name = 'mimic3wdb-matched/1.0'

path_to_numrics_signal_duration = 'data\mimic3wdb-matched_numerics_signals_duration.csv'

def filter_waveform_numerics_by_signals_duration(subjects, signals, min_duration_seconds):
    valid_records = []
    numerics_signal_duration = pd.read_csv(path_to_numrics_signal_duration)

    # prefiltering
    filtered_by_subject = numerics_signal_duration[numerics_signal_duration['subject'].isin(subjects)]
    filtered_by_signals = filtered_by_subject[filtered_by_subject['signal'].isin(signals)]
    filtered_by_duration = filtered_by_signals[filtered_by_signals['duration_seconds'] >= min_duration_seconds]

    # for each record_id check if it has the correct signals
    for record_id in filtered_by_duration['record_id'].unique():
        signals_in_record = filtered_by_duration[filtered_by_duration['record_id'] == record_id]['signal'].unique().tolist()
        if set(signals_in_record) == set(signals):
            valid_records.append({"subject": filtered_by_duration[filtered_by_duration['record_id'] == record_id]['subject'].values[0],
                                "record_id": record_id})
    
    return valid_records

    

def get_all_subjects_mimic_iii_waveform():
    all_subjects = wfdb.get_record_list(database_name)
    return all_subjects

def get_waveform_records_with_signals(subjects, signals):
    records_with_signals = []

    i = 0
    for subject in subjects:
        records = wfdb.get_record_list(f'{database_name}/{subject}')
        for record in records:
            # load header check whether it has all specified signals
            header = wfdb.rdheader(record, pn_dir = f'{database_name}/{subject}', rd_segments=True)
            if all(sig in header.sig_name for sig in signals):
                records_with_signals.append({"record": record, "subject": subject})
        i += 1
        if i % 10 == 0:
            print(f"Processed {i} subjects, found {len(records_with_signals)} records with {signals} so far.")
    
    return records_with_signals


def get_waveform_records_with_signals_and_minimum_recording_duration(subjects, signals, min_duration_seconds=60, numerics_only=True):
    valid_records = []

    i = 0
    for subject in subjects:
        records = wfdb.get_record_list(f'{database_name}/{subject}')
        # sort records by last segment in ascending order
        records = sorted(records, key=lambda x: x.split('_')[-1])
        if numerics_only:
            records = [record for record in records if record.endswith('n')]

        else:
            # remove all records that contain "_"  and records that end with 'n'
            records = [record for record in records if '_' not in record]
            records = [record for record in records if not record.endswith('n')]

        for record in records:
                    # load header check whether it has all specified signals
                    header = wfdb.rdheader(record, pn_dir = f'{database_name}/{subject}', rd_segments=True)
                    if header.sig_len > min_duration_seconds * header.fs:
                        if all(sig in header.sig_name for sig in signals):
                            valid_records.append({"record": record, "subject": subject})
                        
        i += 1
        if i % 100 == 0:
            print(f"Processed {i} subjects, found {len(valid_records)} records with {signals} of {min_duration_seconds / 60} minutes so far.")
    
    return valid_records


################ get statistics about records ########################

def get_number_of_unique_subjects(subject_record_dict_list):
    """
    Get the number of unique subjects in the list of subject_record_dict_list of the format [{"subject": xxx, "record_id": xxxx}]
    """
    # TODO support for txt files
    unique_subjects = set()
    for entry in subject_record_dict_list:
        unique_subjects.add(entry['subject'])
    return len(unique_subjects)

def get_number_of_unique_records(subject_record_dict_list):
    """
    Get the number of unique records in the list of subject_record_dict_list of the format [{"subject": xxx, "record_id": xxxx}]
    """
    # TODO support fot txtx files
    unique_record_ids = set()
    # if entry[0] has a key called 'record_id'
    if 'record_id' in subject_record_dict_list[0]:
        for entry in subject_record_dict_list:
            unique_record_ids.add(entry['record_id'])
    else:
        for entry in subject_record_dict_list:
            unique_record_ids.add(entry['record'])
    return len(unique_record_ids)

def get_number_of_unique_stays(subject_record_dict_list, cur):
    """
    Get the number of unique stays in the list of subject_record_dict_list of the format [{"subject": xxx, "record_id": xxxx}]
    """
    # TODO
    unique_stays = set()
    for entry in subject_record_dict_list:
        subject = entry['subject']
        record_id = entry['record_id'] if 'record_id' in entry else entry['record']
        hadm_id = get_hadm_id_from_record(record_id, cur)
        if hadm_id is not None:
            unique_stays.add((subject, hadm_id))
    return len(unique_stays)

def get_total_duration_of_records(subject_record_dict_list, is_numeric=True):
    """
    Get the total duration of all records in the list of subject_record_dict_list of the format [{"subject": xxx, "record_id": xxxx}]
    """
    total_duration = 0


    if is_numeric:
        records = [entry['record_id'] for entry in subject_record_dict_list]
        # for numeric records, we need to read the duration from the csv file
        numerics_signal_duration = pd.read_csv(path_to_numrics_signal_duration)
        filtered_by_record = numerics_signal_duration[numerics_signal_duration['record_id'].isin(records)]
        filtered = filtered_by_record[['subject', 'record_id', 'duration_seconds']]
        filtered = filtered.drop_duplicates(subset=['subject', 'record_id'])
        total_duration = filtered['duration_seconds'].sum()
    else:
        i = 0
        for entry in subject_record_dict_list:
            record = entry['record_id'] if 'record_id' in entry else entry['record']
            subject = entry['subject']
            header = wfdb.rdheader(record, pn_dir = f'{database_name}/{subject}', rd_segments=True)
            #header = wfdb.rdheader(record, pn_dir=database_name)
            total_duration += header.sig_len / header.fs
            i += 1
            if i % 1000 == 0:
                print(f"Processed {i} records, total duration so far: {total_duration} seconds.")

    return total_duration

def print_statistics_of_waveform_records(subject_record_dict_list, cur):
    unique_subjects = get_number_of_unique_subjects(subject_record_dict_list)
    unique_records = get_number_of_unique_records(subject_record_dict_list)
    unique_stays = get_number_of_unique_stays(subject_record_dict_list, cur)
    total_duration = get_total_duration_of_records(subject_record_dict_list)
    
    print(f"Number of unique subjects: {unique_subjects}")
    print(f"Number of unique records: {unique_records}")
    print(f"Number of unique stays: {unique_stays}")
    print(f"Total duration of records: {total_duration} seconds or {total_duration / 60} minutes or {total_duration / 3600} hours or {total_duration / 86400} days")

def process_subject_chunk(args):
    """
    Process a chunk of subjects for waveform records.
    This function will be called by each process.
    """
    subject_chunk, signals, min_duration_seconds, numerics_only, process_id = args
    
    valid_records = []
    processed_count = 0
    
    print(f"Process {process_id}: Starting to process {len(subject_chunk)} subjects")
    valid_records = get_waveform_records_with_signals_and_minimum_recording_duration(subject_chunk, signals, min_duration_seconds, numerics_only)
    
    print(f"Process {process_id}: Completed. Found {len(valid_records)} valid records from {processed_count} subjects")
    return valid_records

def split_subjects_into_chunks(subjects, num_processes):
    """Split the list of subjects into chunks for each process."""
    chunk_size = len(subjects) // num_processes
    chunks = []
    
    for i in range(num_processes):
        start_idx = i * chunk_size
        if i == num_processes - 1:  # Last process gets remaining subjects
            end_idx = len(subjects)
        else:
            end_idx = (i + 1) * chunk_size
        
        chunks.append(subjects[start_idx:end_idx])
    
    return chunks

def get_waveform_records_with_signals_and_minimum_recording_duration_multiprocess(
    subjects, 
    signals, 
    min_duration_seconds, 
    numerics_only, 
    num_processes
):
    """
    Multiprocessing version of get_waveform_records_with_signals_and_minimum_recording_duration.
    
    Args:
        subjects: List of subjects to process
        signals: Required signals
        min_duration_seconds: Minimum duration requirement
        numerics_only: Whether to process only numeric records
        num_processes: Number of processes to use (defaults to CPU count)
    
    Returns:
        List of valid records
    """

    if num_processes is None:
        num_processes = min(cpu_count(), len(subjects))
    
    print(f"Starting multiprocessing with {num_processes} processes for {len(subjects)} subjects.")
    
    # Split subjects into chunks for each process
    subject_chunks = split_subjects_into_chunks(subjects, num_processes)
    
    # Print chunk sizes for verification
    for i, chunk in enumerate(subject_chunks):
        print(f"Process {i+1} will process {len(chunk)} subjects.")
    
    # Prepare arguments for each process
    print("numerics_only:", numerics_only)
    process_args = []
    for process_id, chunk in enumerate(subject_chunks):
        process_args.append((chunk, signals, min_duration_seconds, numerics_only, process_id + 1))
    
    # Use multiprocessing Pool
    start_time = time.time()
    
    try:
        with Pool(processes=num_processes) as pool:
            print("Starting parallel processing...")
            results = pool.map(process_subject_chunk, process_args)
            
            # Combine results from all processes
            all_valid_records = []
            for result in results:
                all_valid_records.extend(result)
            
    except Exception as e:
        print(f"Error during multiprocessing: {e}")
        return []
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nAll processes completed!")
    print(f"Total valid records found: {len(all_valid_records)}")
    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Average processing rate: {len(subjects)/processing_time:.2f} subjects/second")
    
    return all_valid_records

def process_record_chunk_duration(args):
    """
    Process a chunk of records to calculate their total duration.
    This function will be called by each process.
    """
    record_chunk, is_numeric, process_id = args
    
    total_duration = 0
    processed_count = 0
    
    print(f"Process {process_id}: Starting to process {len(record_chunk)} records")
    
    if is_numeric:
        # For numeric records, use the CSV file approach
        records = [entry['record_id'] if 'record_id' in entry else entry['record'] for entry in record_chunk]
        numerics_signal_duration = pd.read_csv(path_to_numrics_signal_duration)
        filtered_by_record = numerics_signal_duration[numerics_signal_duration['record_id'].isin(records)]
        filtered = filtered_by_record[['subject', 'record_id', 'duration_seconds']]
        filtered = filtered.drop_duplicates(subset=['subject', 'record_id'])
        total_duration = filtered['duration_seconds'].sum()
        processed_count = len(filtered)
        
        print(f"Process {process_id}: Completed (numeric). Total duration: {total_duration/3600:.2f} hours from {processed_count} records")
        
    else:
        # For non-numeric records, read headers directly
        for i, entry in enumerate(record_chunk):
            try:
                record = entry['record_id'] if 'record_id' in entry else entry['record']
                subject = entry['subject']
                
                header = wfdb.rdheader(record, pn_dir=f'{database_name}/{subject}', rd_segments=True)
                duration_seconds = header.sig_len / header.fs
                total_duration += duration_seconds
                processed_count += 1
                
                # Progress update every 50 records per process
                if (i + 1) % 50 == 0:
                    print(f"Process {process_id}: Processed {i + 1}/{len(record_chunk)} records, "
                          f"total duration so far: {total_duration/3600:.2f} hours")
                    
            except Exception as e:
                print(f"Process {process_id}: Error processing record {entry}: {e}")
                continue
        
        print(f"Process {process_id}: Completed. Processed {processed_count} records, "
              f"total duration: {total_duration/3600:.2f} hours")
    
    return {
        'total_duration': total_duration,
        'processed_count': processed_count,
        'process_id': process_id
    }

def split_records_into_chunks(records, num_processes):
    """Split the list of records into chunks for each process."""
    chunk_size = len(records) // num_processes
    chunks = []
    
    for i in range(num_processes):
        start_idx = i * chunk_size
        if i == num_processes - 1:  # Last process gets remaining records
            end_idx = len(records)
        else:
            end_idx = (i + 1) * chunk_size
        
        chunks.append(records[start_idx:end_idx])
    
    return chunks

def get_total_duration_of_records_multiprocess(subject_record_dict_list, is_numeric=True, num_processes=None):
    """
    Multiprocessing version of get_total_duration_of_records.
    
    Args:
        subject_record_dict_list: List of record dictionaries with 'record_id'/'record' and 'subject' keys
        is_numeric: Whether records are numeric (uses CSV for faster processing)
        num_processes: Number of processes to use (defaults to CPU count)
    
    Returns:
        Dictionary with total duration and statistics
    """
    if num_processes is None:
        num_processes = min(cpu_count(), len(subject_record_dict_list))
    
    print(f"Starting multiprocessing duration calculation with {num_processes} processes for {len(subject_record_dict_list)} records.")
    
    # For numeric records with small datasets, just use the original function
    if is_numeric and len(subject_record_dict_list) < 100:
        print("Small dataset detected, using single-threaded approach for numeric records.")
        total_duration = get_total_duration_of_records(subject_record_dict_list, is_numeric)
        return {
            'total_duration_seconds': total_duration,
            'total_duration_hours': total_duration / 3600,
            'total_duration_days': total_duration / 86400,
            'total_records_processed': len(subject_record_dict_list),
            'average_duration_per_record_hours': (total_duration / len(subject_record_dict_list)) / 3600,
            'processing_time_seconds': 0,
            'processing_rate_records_per_second': float('inf')
        }
    
    # Split records into chunks for each process
    record_chunks = split_records_into_chunks(subject_record_dict_list, num_processes)
    
    # Print chunk sizes for verification
    for i, chunk in enumerate(record_chunks):
        print(f"Process {i+1} will process {len(chunk)} records.")
    
    # Prepare arguments for each process
    process_args = []
    for process_id, chunk in enumerate(record_chunks):
        process_args.append((chunk, is_numeric, process_id + 1))
    
    # Use multiprocessing Pool
    start_time = time.time()
    
    try:
        with Pool(processes=num_processes) as pool:
            print("Starting parallel processing...")
            results = pool.map(process_record_chunk_duration, process_args)
            
            # Combine results from all processes
            total_duration = 0
            total_processed = 0
            
            for result in results:
                total_duration += result['total_duration']
                total_processed += result['processed_count']
                print(f"Process {result['process_id']}: {result['total_duration']/3600:.2f} hours "
                      f"from {result['processed_count']} records")
            
    except Exception as e:
        print(f"Error during multiprocessing: {e}")
        return None
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Calculate statistics
    total_hours = total_duration / 3600
    total_days = total_hours / 24
    avg_duration_per_record = total_duration / total_processed if total_processed > 0 else 0
    
    print(f"\n" + "="*60)
    print(f"TOTAL DURATION STATISTICS")
    print(f"="*60)
    print(f"Total records processed: {total_processed}")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Total duration: {total_hours:.2f} hours")
    print(f"Total duration: {total_days:.2f} days")
    print(f"Average duration per record: {avg_duration_per_record/3600:.2f} hours")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Processing rate: {total_processed/processing_time:.2f} records/second")
    print(f"="*60)
    
    return {
        'total_duration_seconds': total_duration,
        'total_duration_hours': total_hours,
        'total_duration_days': total_days,
        'total_records_processed': total_processed,
        'average_duration_per_record_hours': avg_duration_per_record/3600,
        'processing_time_seconds': processing_time,
        'processing_rate_records_per_second': total_processed/processing_time if processing_time > 0 else 0
    }


##################################

def transtlate_txt_to_csv_with_start_and_end(path_to_txt, output_csv_path):
    """
    Translates a text file with records into a CSV file with start and end times.
    :param path_to_txt: Path to the input text file.
    :param output_csv_path: Path to the output CSV file.
    """

    # read records
    records = []
    with open(path_to_txt, "r") as f:
        lines = f.readlines()
        for line in lines:
            records.append(line.strip())

    no_ventilation_df = pd.DataFrame(records, columns=["record"])
    no_ventilation_df["offset_start_seconds"] = 0



    path_to_numrics_signal_duration = 'data\mimic3wdb-matched_numerics_signals_duration.csv'
    numerics_signal_duration = pd.read_csv(path_to_numrics_signal_duration)
    filtered_by_record = numerics_signal_duration[numerics_signal_duration['record_id'].isin(no_ventilation_df['record'])]
    # only keep first entry per record_id
    filtered_by_record = filtered_by_record.groupby('record_id').first().reset_index()

    record_duration_df = filtered_by_record[['record_id', 'duration_seconds']].copy()

    # concat with no_ventilation_df on record / record_id
    combined_df = pd.merge(no_ventilation_df, record_duration_df, left_on='record', right_on='record_id', how='left')
    combined_df["offset_end_seconds"] = combined_df["duration_seconds"]
    combined_df.drop(columns=['record_id', 'duration_seconds'], inplace=True)

    combined_df.to_csv(output_csv_path, index=False)




#####################################################
def check_number_of_icu_stays(records, cur):
    icu_stay_counts = {}
    for record in records:
        record_id = record['record_id']
        subject_id = int(record_id.split('-')[0].replace('p', ''))
        hadm_id = get_hadm_id_from_record(record_id, cur)
        cur.execute("SELECT COUNT(*) FROM mimiciii.icustays WHERE hadm_id=%s", (hadm_id,))
        icu_stay_count = cur.fetchone()[0]
        icu_stay_counts[record_id] = icu_stay_count
    return icu_stay_counts