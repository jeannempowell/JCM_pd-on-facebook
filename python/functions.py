import requests
import pandas as pd
import io
from constants import url
import json
import numpy as np
import re
from redcap import Project
from constants import posts_token
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import string
from nltk.tokenize import word_tokenize
# Download only if stopwords are missing

try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))
except:
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))


import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.sparse import hstack

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, recall_score, accuracy_score, precision_score, f1_score
import joblib
import time






def pull_redcap_report(token, report_id):
    """
    Pulls a REDCap report using the provided report ID and returns a pandas DataFrame.

    Parameters:
    report_id (str): The ID of the REDCap report to pull.

    Returns:
    pandas.DataFrame: The data from the REDCap report.
    """
    formData = {
        'token': token,
        'content': 'report',
        'format': 'csv',
        'report_id': report_id,
        'csvDelimiter': '',
        'rawOrLabel': 'raw',
        'rawOrLabelHeaders': 'raw',
        'exportCheckboxLabel': 'false',
        'returnFormat': 'json'
    }

    # Download data from REDCap
    response = requests.post(url, data=formData)
    response.raise_for_status()  # Raise an HTTPError if the response contains an HTTP error status code

    data = response.content.decode('utf-8')
    redcap = pd.read_csv(io.StringIO(data))

    return redcap

def load_data(file_data, record_id, json_file):
  # This function 'load_data' is designed to load files from REDCap using the API.
  # It takes two parameters: 'record_id' (the ID of the record) and 'json_file' (the field name of the file).
  
  # Update the 'redcap_data' dictionary with the specific record ID and JSON file field name.
  file_data['record'] = record_id
  file_data['field'] = json_file

  # Perform a POST request to the REDCap API with the updated 'redcap_data'.
  r = requests.post('https://redcap.emory.edu/api/', data=file_data)

  # The response is expected to be a JSON-formatted string.
  # The function parses this string into a Python object (dictionary or list).
  # If there's no file to download, 'r.text' will contain an error message.
  return json.loads(r.text) 


def extract_text(record_id, df_text, json_file, json_data):
    group = np.nan

    # Count the number of entries in the JSON file
    num_entries = len(json_data)
    
    # Count the initial number of rows in df_text
    initial_rows = len(df_text)

    # List to store entries that were not added to df_text
    not_added_entries = []


    for entry in json_data:
        entry_added = False  # Flag to track if the entry was added

        if json_file in ['anonymous_posts_you_ve_written']:
            try:
                timestamp = entry.get('timestamp', np.nan)
                title = np.nan
                group = 'unknown group'
                if 'data' in entry and isinstance(entry['data'], list) and len(entry['data']) > 0:
                    text = entry['data'][0].get('post', np.nan)
                
                if isinstance(text, str):

                    df_text.loc[len(df_text)] = [record_id, json_file, timestamp, title, text, group]
                    entry_added = True

            except (KeyError, TypeError) as e:
                print(f"Error processing entry: {entry}")
                print(f"Exception: {e}")
                

        elif json_file in ['your_posts_check_ins_photos_and_videos', 'archive', 'group_posts_and_comments', 'your_posts', 'trash_json', 'your_pending_posts_in_groups']:
            try:
                timestamp = entry.get('timestamp', np.nan)
                title = entry.get('title', np.nan)

                # Extract 'post' text if available
                if 'data' in entry and isinstance(entry['data'], list) and len(entry['data']) > 0:
                    text = entry['data'][0].get('post', np.nan)
                else:
                    text = np.nan

                # Extract group information for 'group_posts_and_comments'
                if isinstance(text, str):
                    if json_file == 'group_posts_and_comments' and isinstance(title, str):
                        match = re.search(r'(?<=\bin\b\s)(.*?)(?=\.)', title)
                        group = match.group(0) if match else np.nan
                    
                    elif json_file == 'your_pending_posts_in_groups':
                        group = 'Unknown'
                    else:
                        group = np.nan

                    df_text.loc[len(df_text)] = [record_id, json_file, timestamp, title, text, group]
                    entry_added = True

                # Extract 'description' from 'media' if available in attachments
                if 'attachments' in entry and isinstance(entry['attachments'], list):
                    for attachment in entry['attachments']:
                        if 'data' in attachment and isinstance(attachment['data'], list):
                            for data_item in attachment['data']:
                                # Extract 'media' description if available
                                if 'media' in data_item and isinstance(data_item['media'], dict):
                                    description = data_item['media'].get('description', np.nan)
                                    #creation_timestamp = data_item['media'].get('creation_timestamp', np.nan) # removed this so that each entry has the same timestamp even if multiple attachment/data subentries within the entry
                                    if isinstance(description, str):
                                        df_text.loc[len(df_text)] = [record_id, json_file, timestamp, title, description, group]
                                        entry_added = True
                                
                                # Extract 'text' from 'attachments -> data'
                                text = data_item.get('text', np.nan)
                                if isinstance(text, str):
                                    df_text.loc[len(df_text)] = [record_id, json_file, timestamp, title, text, group]
                                    entry_added = True

            except (KeyError, TypeError) as e:
                print(f"Error processing entry: {entry}")
                print(f"Exception: {e}")

        # Extract information for 'your_uncategorized_photos', 'your_videos'
        elif json_file in ['your_uncategorized_photos', 'your_videos']:
            title = np.nan
            try:
                timestamp = entry.get('creation_timestamp', np.nan)
                text = entry.get('description', np.nan)
                title = entry.get('title', np.nan)

                if isinstance(text, str):
                    df_text.loc[len(df_text)] = [record_id, json_file, timestamp, title, text, group]
                    entry_added = True
            
            except (KeyError, TypeError) as e:
                print(f"Error processing entry: {entry}")
                print(f"Exception: {e}")

        # Extract information for 'album'
        elif json_file in ['album']:
            try:
                title = 'Album title'
                timestamp = entry.get('last_modified_timestamp', np.nan)
                album_name = entry.get('name', np.nan)

                if album_name not in [np.nan, 'Untitled album', 'Profile pictures', 'Cover photos', 'Timeline photos', 'Instagram Photos', 'Mobile Uploads']:
                    df_text.loc[len(df_text)] = [record_id, json_file, timestamp, title, album_name, group]
                    entry_added = True
            
            except (KeyError, TypeError) as e:
                print(f"Error processing entry: {entry}")
                print(f"Exception: {e}")
            
            title = f'{album_name} photo description'

            try:
                cover_photo = entry.get('cover_photo', {})
                if cover_photo:
                    timestamp = cover_photo.get('creation_timestamp', np.nan)
                    text = cover_photo.get('description', np.nan)

                    if isinstance(text, str):
                        df_text.loc[len(df_text)] = [record_id, json_file, timestamp, title, text, group]
                        entry_added = True
                        
            except (KeyError, TypeError) as e:
                print(f"Error processing cover photo: {cover_photo}")
                print(f"Exception: {e}")
            
            for photo in entry.get('photos', []):
                try:
                    timestamp = photo.get('creation_timestamp', np.nan)
                    text = photo.get('description', np.nan)
                    
                    if isinstance(text, str):
                        df_text.loc[len(df_text)] = [record_id, json_file, timestamp, title, text, group]
                        entry_added = True
                
                except (KeyError, TypeError) as e:
                    print(f"Error processing photo: {photo}")
                    print(f"Exception: {e}")
    
        # Extract information for 'comments', 'your_comments_in_groups'
        elif json_file in ['comments', 'your_comments_in_groups']:
            try:
                timestamp = entry.get('timestamp', np.nan)
                text = entry.get('data', [{}])[0].get('comment', {}).get('comment', np.nan)
                group = entry.get('data', [{}])[0].get('comment', {}).get('group', np.nan)
                title = entry.get('title', np.nan)

                if isinstance(text, str):
                    df_text.loc[len(df_text)] = [record_id, json_file, timestamp, title, text, group]
                    entry_added = True
                
            except (KeyError, TypeError) as e:
                print(f"Error processing album: {entry}")
                print(f"Exception: {e}")
        
        elif json_file in ['your_answers_to_membership_questions']:
            try:
                title = "Answers to group membership questions"
                group = entry.get('group_name', np.nan)
                for subentry in entry['answers']:
                    text = subentry.get('answer', np.nan)
                    timestamp = subentry.get('timestamp', np.nan)

                    if isinstance(text, str):
                        df_text.loc[len(df_text)] = [record_id, json_file, timestamp, title, text, group]
                        entry_added = True
            except (KeyError, TypeError) as e:
                    print(f"Error processing entry: {entry}")
                    print(f"Exception: {e}")


        # If entry was not added, append it to not_added_entries list
        if not entry_added:
            not_added_entries.append(entry)
 
    # Count the final number of rows in df_text
    final_rows = len(df_text)
    
    # Calculate the number of new rows added
    new_rows_added = final_rows - initial_rows
    
    # Report the difference
    #print(f"JSON file '{json_file}' contained {num_entries} entries. {new_rows_added} new rows were added to df_text.")
    
    # Write not added entries to a JSON file
    #output_json_file = f'{json_file}.json'
    #if not_added_entries:
    #    with open(output_json_file, 'w') as f:
    #        json.dump(not_added_entries, f, indent=4)
    #    print(f"Entries not added to df_text have been saved to '{output_json_file}'.")

    return df_text



def process_json_files(record_data, redcap_data, file_data, record_id, df_text):
    """
    Processes multiple JSON files, loads their data, applies special handling, 
    and extracts relevant text.

    Parameters:
        record_data (pd.DataFrame): DataFrame containing JSON references.
        redcap_data (pd.DataFrame): DataFrame containing additional metadata.
        file_data (dict or str): The file path or data structure.
        record_id (str): The record identifier.
        df_text (pd.DataFrame): DataFrame to store extracted text.

    Returns:
        pd.DataFrame: Updated df_text with extracted text.
    """

    # List of JSON files to process with optional nested keys
    json_files = {
        "your_posts_check_ins_photos_and_videos": None,
        "your_posts": None,
        "archive": "archive_v2",
        "facebook_editor": "facebook_editor",
        "news_articles_you_ve_opened": None,
        "photo_effects": None,
        "trash_json": "trash_v2",
        "your_sent_shared_album_inv": None,
        "your_received_shared_album_invites": None,
        "your_uncategorized_photos": "other_photos_v2",
        "your_videos": "videos_v2",
        "album": None,
        "comments": "comments_v2",
        "likes_and_reactions": None,  # Special handling below
        "your_comment_active_days": None,
        "anonymous_posts_you_ve_written": "anonymous_author_posts",
        "group_posts_and_comments": "group_posts_v2",
        "your_answers_to_membership_questions": ["group_membership_questions_answers_v2", "group_answers"],
        "your_badges": None,
        "your_comments_in_groups": "group_comments_v2",
        "your_contributions": None,
        "your_group_membership_activity": None,
        "your_groups": None,
        "your_pending_posts_in_groups": "pending_posts_v2",
        "join_requests": None,
        "group_invites_you_ve_sent": None,
        "chat_invites_received": None,
        "your_group_moments": None
    }

    # Special handling for specific JSON files
    special_cases = {
        "your_posts_check_ins_photos_and_videos": {"check_combined": "your_posts_combined", "convert_to_list": ["92"]},
        "album": {
            "convert_to_list": ["92", "386", "388"],
            "flatten_list": ["159"]
        },
        "likes_and_reactions": {"check_combined": "likes_combined"}  # Requires likes_combined check
    }

    for json_file, nested_keys in json_files.items():
        json_value = record_data.iloc[0][json_file]

        if len(json_value) > 0:
            json_data = load_data(file_data, record_id, json_file)

            # Handle special cases
            if json_file in special_cases:
                case = special_cases[json_file]

                # Special case for `your_posts_check_ins_photos_and_videos`
                if "check_combined" in case and json_file == "your_posts_check_ins_photos_and_videos":
                    posts_combined = redcap_data.loc[redcap_data['record_id'] == record_id, case["check_combined"]].iloc[0]

                    # Convert to list if record_id is in convert_to_list
                    if "convert_to_list" in case and record_id in case["convert_to_list"]:
                        json_data = [json_data]

                    if posts_combined == "0":
                        df_text = extract_text(record_id, df_text, json_file, json_data)
                    elif posts_combined == "1":
                        for entry in json_data:
                            df_text = extract_text(record_id, df_text, json_file, entry)
                    continue  # Skip the rest of the processing for this file

                # Convert to list if record_id is in convert_to_list
                if "convert_to_list" in case and record_id in case["convert_to_list"]:
                    json_data = [json_data]

                # Flatten lists for specific records
                if "flatten_list" in case and record_id in case["flatten_list"]:
                    flattened_data = []
                    for element in json_data:
                        if isinstance(element, list):
                            flattened_data.extend(element)
                        else:
                            flattened_data.append(element)
                    json_data = flattened_data

                # Handle `likes_and_reactions` special case with `likes_combined`
                if "check_combined" in case and json_file == "likes_and_reactions":
                    combined_value = record_data.iloc[0][case["check_combined"]]
                    # Additional processing for likes_and_reactions if necessary

            # Extract text if applicable
            if nested_keys:
                if isinstance(nested_keys, list):  # Handle multi-level nested keys
                    nested_data = json_data
                    for key in nested_keys:
                        nested_data = nested_data.get(key, {})
                    df_text = extract_text(record_id, df_text, json_file, nested_data)
                else:
                    df_text = extract_text(record_id, df_text, json_file, json_data[nested_keys])
            else:
                df_text = extract_text(record_id, df_text, json_file, json_data)

    return df_text


def clean_and_deduplicate_text(df):
    """
    Cleans and deduplicates the dataframe by:
    - Removing exact duplicates based on timestamp and text.
    - Dropping rows with missing or empty text.
    - Converting timestamps to datetime.
    - Removing near-duplicate texts within 180 seconds.
    - Filtering out irrelevant posts based on regex patterns.

    Parameters:
        df (pd.DataFrame): The input dataframe containing 'timestamp', 'text', and 'title'.

    Returns:
        pd.DataFrame: The cleaned and deduplicated dataframe.
    """

    # Drop duplicates with exact matches for timestamp and text
    df = df.drop_duplicates(subset=['timestamp', 'text'])

    # Drop rows where 'text' is NaN or empty
    df = df[df['text'].notna() & (df['text'] != '')]

    # Ensure timestamp is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Sort by text and timestamp to ensure the oldest entry is first within each text group
    df.sort_values(by=['text', 'timestamp'], inplace=True)

    # Function to filter out near-duplicates based on a 180-second threshold
    def filter_duplicates(df):
        to_keep = []
        grouped = df.groupby('text')

        for _, group in grouped:
            if len(group) > 1:
                last_kept_index = group.index[0]
                to_keep.append(last_kept_index)

                for i in range(1, len(group)):
                    current_index = group.index[i]
                    time_difference = (group.loc[current_index, 'timestamp'] - 
                                       group.loc[last_kept_index, 'timestamp']).total_seconds()

                    if time_difference > 180:
                        to_keep.append(current_index)
                        last_kept_index = current_index
            else:
                to_keep.append(group.index[0])  # Keep single-entry groups

        return df.loc[to_keep]

    # Apply duplicate filtering
    df = filter_duplicates(df)

    # Define regex pattern for filtering out "shared a memory" posts
    date_pattern = r'[A-Z][a-z]{2} \d{1,2}, \d{4} \d{1,2}:\d{2}:\d{2}(?:am|pm)'


# Filter out rows where 'title' contains 'shared a memory' and 'text' matches any of the specified patterns
    df =df[
        ~(
            df['title'].str.contains('shared a memory', case=False, na=False) &
            (
                df['text'].str.contains(r'\d+ Years Ago', case=False, na=False) |
                df['text'].str.contains(date_pattern, case=False, na=False) |
                df['text'].str.contains('added a new photo', case=False, na=False)
            )
        )
    ]

    # Reset index and sort by timestamp
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    return df


def safe_decode(text):
    try:
        # Ensure text is a string and apply the regex
        if isinstance(text, str):
            
            return re.sub(
                r'[\xc2-\xf4][\x80-\xbf]+',  # Regex for UTF-8 multibyte characters
                lambda m: m.group(0).encode('latin1').decode('utf8'),
                text
            )
        else:
            return text  # If not a string, return as-is
    except (UnicodeEncodeError, UnicodeDecodeError):
        print(text)
        return text  # Return original text if decoding fails
    

def redcap_upload(data):
    """
    Uploads prepared data to REDCap.

    Parameters:
        data (pd.DataFrame): DataFrame containing data to upload. Must include 'record_id'.

    Returns:
        response (dict): Response from the REDCap API.
    """
    # Ensure 'record_id' is the first column
    cols = ['record_id'] + [col for col in data.columns if col != 'record_id']
    data = data[cols]

    # Convert DataFrame to a list of dictionaries for REDCap upload
    import_list = []
    for x in data.index:
        # Initialize a dictionary for the current record
        entry = {}

        # Iterate over each column in the dataframe
        for column in data.columns:
            value = data.at[x, column]
            
            # Handle lists (if necessary)
            if type(value) == list:
                value = ", ".join(value)
            
            # Add the column value to the dictionary
            entry[column] = str(value)
            if entry[column] == 'None': entry[column] = ""
            if pd.isna(entry[column]): entry[column] = ""
            if entry[column] == 'nan': entry[column] = ""
        
        # Add the dictionary to the import list
        import_list.append(entry)

    # Import data into REDCap using the PyCap library
    project = Project(url, posts_token)
    response = project.import_records(import_list)

    # Log imported records
    print('Import complete.')

    return response

def preprocess_text(df, text_column):
    """
    This function preprocesses the text by converting it to lowercase, removing punctuation, 
    and applying stemming to each word in the text.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the text data.
    text_column (str): The name of the column in the dataframe that contains the text to be processed.

    Returns:
    pandas.DataFrame: The dataframe with processed text in the specified column.
    """
    # Create a copy of the dataframe to avoid modifying the original dataframe
    df1 = df.copy()

    # Initialize the tokenizer and stemmer
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()

    # Define a function to preprocess a single text string
    def preprocess_string(text):
        # Convert text to lowercase and tokenize (removing punctuation)
        tokens = tokenizer.tokenize(text.lower())
        # Apply stemming to each token
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        # Join tokens back into a string
        return ' '.join(stemmed_tokens)

    # Apply the preprocess function to each entry in the specified column
    df1[text_column] = df1[text_column].astype(str).apply(preprocess_string)

    return df1
def remove_stopwords(df1, text_column):
    """
    This function removes stopwords from the text in the specified dataframe column.

    Parameters:
    df1 (pandas.DataFrame): The dataframe containing the text data.
    text_column (str): The name of the column in the dataframe from which stopwords are to be removed.

    Returns:
    pandas.DataFrame: The dataframe with stopwords removed from the specified column.
    """
    # Create a copy of the dataframe to avoid modifying the original dataframe
    df = df1.copy()

    # Get the set of English stopwords
    stop_words = set(stopwords.words('english'))

    # Define a function to remove stopwords from a single text string
    def remove_stopwords_string(string):
        # Split the string into words, remove stopwords, and rejoin into a single string
        return ' '.join([word for word in string.lower().split() if word not in stop_words])

    # Apply the function to each entry in the specified column
    df[text_column] = df[text_column].astype(str).apply(remove_stopwords_string)

    return df
def preprocess_terms(term_dictionary, stem = True):
    """
    This function preprocesses the terms in the term_dictionary by converting them to lowercase, 
    removing punctuation, and applying stemming to each word in the terms.

    Parameters:
    term_dictionary (dict): Dictionary of term lists to be processed.

    Returns:
    dict: Dictionary with the same structure but preprocessed terms.
    """
    # Initialize the tokenizer and stemmer
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()

    # Preprocess each term in the lists associated with each key
    preprocessed_dict = {}
    for key, terms in term_dictionary.items():
        if stem: preprocessed_terms = [' '.join(stemmer.stem(word) for word in tokenizer.tokenize(term.lower())) for term in terms]
        else: preprocessed_terms = [' '.join(word for word in tokenizer.tokenize(term.lower())) for term in terms] 
        preprocessed_dict[key] = preprocessed_terms

    return preprocessed_dict


def search_terms_in_text(df, term_dictionary1, term_dictionary2):
    """
    Searches for instances of terms in the preprocessed text column of the DataFrame and
    flags the keywords and their associated categories.

    Parameters:
    df (pandas.DataFrame): DataFrame with preprocessed text.
    term_dictionary1 (dict): Dictionary of terms to search for in the original text.
    term_dictionary2 (dict): Dictionary of terms to search for in the processed text.

    Returns:
    pandas.DataFrame: DataFrame with columns for flagged keywords and their categories.
    """
    # Initialize columns for flagged keywords and categories
    df['keywords'] = ''
    df['keyword_categories'] = ''

    # Function to find keywords and categories in a given text
    def find_keywords_and_categories(text, term_dict):
        found_keywords = set()
        found_categories = set()
        for key, terms in term_dict.items():
            for term in terms:
                if len(term) > 9:  # Direct match for longer terms
                    if term in text:
                        found_keywords.add(term)
                        found_categories.add(key)
                elif ' ' in term:  # Direct match for multi-word terms
                    if term in text:
                        found_keywords.add(term)
                        found_categories.add(key)
                else:  # Word boundary match for shorter terms
                    if re.search(r'\b' + re.escape(term) + r'\b', text):
                        found_keywords.add(term)
                        found_categories.add(key)
        return found_keywords, found_categories

    # Process each row in the DataFrame
    for index, row in df.iterrows():
        original_text = str(row['text'])
        processed_text = row['text_processed']
        
        # Find keywords and categories in both original and processed text
        keywords_in_original, categories_in_original = find_keywords_and_categories(original_text, term_dictionary1)
        keywords_in_processed, categories_in_processed = find_keywords_and_categories(processed_text, term_dictionary2)
        
        # Combine keywords and categories from original and processed text
        all_keywords = keywords_in_original.union(keywords_in_processed)
        all_categories = categories_in_original.union(categories_in_processed)
        
        # Add results to the DataFrame
        df.at[index, 'keywords'] = ', '.join(all_keywords) if all_keywords else ''
        df.at[index, 'keyword_categories'] = ', '.join(all_categories) if all_categories else ''

    return df




def get_age_at_post(posts, participants):
    """
    Merges participant information (dob, gender, pd_yesno) into posts,
    ensures date formats are correct, and calculates age at post.

    Parameters:
        posts (pd.DataFrame): DataFrame containing social media posts.
        participants (pd.DataFrame): DataFrame containing participant metadata.

    Returns:
        pd.DataFrame: Updated posts DataFrame with merged participant info and age_at_post.
    """
    # Ensure 'participant_id' is consistent
    participants['participant_id'] = participants['record_id']

    # Merge relevant columns
    posts = posts.merge(
        participants[['participant_id', 'dob', 'gender', 'pd_yesno']],
        on='participant_id',
        how='left'
    )

    # Convert dates to datetime format
    posts['dob'] = pd.to_datetime(posts['dob'], errors='coerce')  # Handle invalid/missing dates
    posts['timestamp'] = pd.to_datetime(posts['timestamp'], errors='coerce')

    # Calculate age at post safely (handling missing values)
    posts['age_at_post'] = posts.apply(
        lambda row: row['timestamp'].year - row['dob'].year - 
                    ((row['timestamp'].month, row['timestamp'].day) < (row['dob'].month, row['dob'].day))
        if pd.notna(row['dob']) and pd.notna(row['timestamp']) else pd.NA,
        axis=1
    )

    return posts



def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)  # Remove mentions and hashtags
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return " ".join(tokens)

def load_word_clusters(file_path):
    word_clusters = {}
    with open(file_path) as f:
        for line in f:
            items = line.strip().split()
            word_clusters[items[1]] = items[0]
    return word_clusters

word_clusters = load_word_clusters('../word_clusters/50mpaths2.txt')

def get_cluster_features(text):
        tokens = word_tokenize(text)
        cluster_string = ''
        for token in tokens:
            if token in word_clusters:
                cluster_string += f"clust_{word_clusters[token]}_clust "
        return cluster_string.strip()



def preprocess_features(train_data, test_data, save_path="../classifiers-2-2-25"):
    """
    Prepares feature matrices for training and testing.

    Steps:
    1. Vectorizes text features (TF-IDF).
    2. Scales numerical features.
    3. Encodes categorical features.
    4. Stacks all features into a final matrix.

    Parameters:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        save_path (str): Directory to save vectorizers and scalers.

    Returns:
        X_train (scipy.sparse matrix): Feature matrix for training.
        X_test (scipy.sparse matrix): Feature matrix for testing.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
    """

    # TF-IDF Vectorization
    cluster_vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=2)
    lemma_vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2)

    X_train_cluster = cluster_vectorizer.fit_transform(train_data['text_cluster'])
    X_train_lemma = lemma_vectorizer.fit_transform(train_data['lemma'])

    X_test_cluster = cluster_vectorizer.transform(test_data['text_cluster'])
    X_test_lemma = lemma_vectorizer.transform(test_data['lemma'])

    # Save vectorizers
    joblib.dump(lemma_vectorizer, f"{save_path}/lemma_vectorizer.joblib")
    joblib.dump(cluster_vectorizer, f"{save_path}/cluster_vectorizer.joblib")
    print("Vectorizers saved.")

    # Numerical Feature Scaling
    scaler = MinMaxScaler()
    X_train_age = scaler.fit_transform(train_data[['age_at_post']])
    X_test_age = scaler.transform(test_data[['age_at_post']])

    # Save the scaler
    joblib.dump(scaler, f"{save_path}/scaler.joblib")
    print("Scaler saved.")

    # Categorical Encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_cat = encoder.fit_transform(train_data[['gender', 'pd_yesno']])
    X_test_cat = encoder.transform(test_data[['gender', 'pd_yesno']])

    # Save the encoder
    joblib.dump(encoder, f"{save_path}/encoder.joblib")
    print("Encoder saved.")

    # Stack all features together
    X_train = hstack([X_train_cluster, X_train_lemma, X_train_age, X_train_cat])
    X_test = hstack([X_test_cluster, X_test_lemma, X_test_age, X_test_cat])

    # Extract target labels
    y_train = train_data['manual_label_pd_relevant']
    y_test = test_data['manual_label_pd_relevant']

    return X_train, X_test, y_train, y_test


# Define a macro recall scoring function
macro_recall_scorer = make_scorer(recall_score, average='macro')

def train_and_evaluate_model(model, param_grid, X_train, y_train, X_test, y_test, model_name, test_predictions):
    """
    Train and evaluate a model using GridSearchCV, optimize for macro recall, 
    save the best model, and collect predictions.

    Parameters:
        model (sklearn model): The classifier to train.
        param_grid (dict): Hyperparameter grid for tuning.
        X_train, y_train: Training data and labels.
        X_test, y_test: Test data and labels.
        model_name (str): Name of the model.

    Returns:
        best_model: The best model found via GridSearchCV.
    """
    start_time = time.time()
    print(f"Training {model_name}...")

    # GridSearchCV setup for macro recall
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=macro_recall_scorer,
        cv=StratifiedKFold(n_splits=5),
        n_jobs=-1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    print(f"Finished training {model_name} in {elapsed_time:.2f} seconds.\n")

    # Get the best model
    best_model = grid_search.best_estimator_
    print(f"Best hyperparameters for {model_name}: {grid_search.best_params_}")

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Save predictions
    test_predictions[f"{model_name.lower()}_test_pred"] = y_pred
    
    # Save the trained model
    model_filename = f"{model_name}_best_model.joblib"
    joblib.dump(best_model, model_filename)
    print(f"Saved {model_name} to {model_filename}.\n")

    return best_model


def evaluate(preds, labels, metric, average=None):
    """Calculate a performance metric."""
    if metric == 'accuracy':
        return accuracy_score(labels, preds)
    elif metric == 'f1':
        return f1_score(labels, preds, average=average)
    elif metric == 'recall':
        return recall_score(labels, preds, average=average)
    elif metric == 'precision':
        return precision_score(labels, preds, average=average)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def calculate_confidence_interval(preds, labels, metric, average=None, num_simulations=1000):
    """Calculate 95% confidence intervals for a performance metric."""
    pred_gold_pairs = [(label, pred) for pred, label in zip(preds, labels)]
    pair_indices = list(range(len(pred_gold_pairs)))

    simulations = []

    for _ in range(num_simulations):
        sampled_indices = np.random.choice(pair_indices, size=len(pred_gold_pairs), replace=True)
        sampled_pairs = [pred_gold_pairs[i] for i in sampled_indices]
        sampled_labels, sampled_preds = zip(*sampled_pairs)

        score = evaluate(sampled_preds, sampled_labels, metric, average=average)
        simulations.append(score)

    simulations = sorted(simulations)
    lower = simulations[int(num_simulations * 0.025)]
    upper = simulations[int(num_simulations * 0.975)]
    return f"({round(lower, 2)}-{round(upper, 2)})"

def generate_results_with_ci_from_columns(test_predictions, y_test, num_simulations=1000):
    """Generate performance metrics and confidence intervals for each model using columns."""
    results = []

    for column in test_predictions.columns:
        if column == 'record_id':
            continue  # Skip the record_id column

        print(f"Processing {column}...")

        y_pred = test_predictions[column]

        # Calculate metrics and confidence intervals
        metrics = ['accuracy', 'recall', 'precision', 'f1']
        averages = ['macro']  # Optimizing for macro recall
        classes = [0, 1]  # Specific classes

        for metric in metrics:
            for average in averages:
                score = evaluate(y_pred, y_test, metric, average=average)
                ci = calculate_confidence_interval(y_pred, y_test, metric, average=average, num_simulations=num_simulations)
                results.append({
                    'Model': column,
                    'Metric': metric,
                    'Type': f"{average.capitalize()}",
                    'Score': round(score, 2),
                    '95% CI': ci
                })

            # For per-class metrics
            if metric != 'accuracy':  # Accuracy is not calculated per class
                for cls in classes:
                    cls_pred = np.array([1 if p == cls else 0 for p in y_pred])
                    cls_labels = np.array([1 if l == cls else 0 for l in y_test])
                    score = evaluate(cls_pred, cls_labels, metric, average=None)
                    ci = calculate_confidence_interval(cls_pred, cls_labels, metric, average=None, num_simulations=num_simulations)
                    results.append({
                        'Model': column,
                        'Metric': metric,
                        'Type': f"Class {cls}",
                        'Score': round(score, 2),
                        '95% CI': ci
                    })

    return pd.DataFrame(results)