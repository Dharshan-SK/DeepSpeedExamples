import boto3
import pandas as pd
import numpy as np
from collections import defaultdict

OCR_BUCKET = "javis-ai-parser-dev"


def download_from_s3(filename, bucket_name, s3_filename, s3_client=None):
    if not s3_client:
        s3_client = boto3.client('s3')
    s3_client.download_file(bucket_name, s3_filename, filename)
    print(f'Downloaded file {s3_filename} from S3 bucket {bucket_name}')

def get_ocr_data_for_doc(doc):
    ocr_s3_path = f'ocr_output/{doc}.parquet'
    ocr_filename = ocr_s3_path.split('/')[-1]
    ocr_local_path = f'/tmp/{ocr_filename}'
    download_from_s3(ocr_local_path, OCR_BUCKET, ocr_s3_path)
    word_df = pd.read_parquet(ocr_local_path)

    return word_df

def filter_out_of_page_elements(page_ocr_data):

    page_ocr_data = page_ocr_data[(page_ocr_data['minx']>=0) & (page_ocr_data['miny']>=0) & 
                                  (page_ocr_data['maxx']<=1) & (page_ocr_data['maxy']<=1)]
    
    return page_ocr_data


def filter_zero_width_height_elements(page_ocr_data):
    # For 20226811734211500, 202262211921390577
    page_ocr_data = page_ocr_data[(page_ocr_data['maxx'] - page_ocr_data['minx'] > 0) & (page_ocr_data['maxy'] - page_ocr_data['miny'] > 0)]
    return page_ocr_data

def get_processed_page_ocr_data(ocr_data, page_number):
    page_ocr_data = ocr_data[ocr_data['page'] == page_number]
    page_ocr_data = filter_out_of_page_elements(page_ocr_data)
    page_ocr_data = filter_zero_width_height_elements(page_ocr_data)
    
    return page_ocr_data

def ocr_to_aligned_text_no_watermark(page_ocr_data, size_threshold=1.5, tilt_threshold=0.1, span_threshold=0.5):
    # Calculate document dimensions
    doc_width = page_ocr_data['maxx'].max() - page_ocr_data['minx'].min()
    doc_height = page_ocr_data['maxy'].max() - page_ocr_data['miny'].min()

    # Filter out potential watermark text
    def is_not_watermark(word):
        word_width = word['maxx'] - word['minx']
        word_height = word['maxy'] - word['miny']
        word_tilt = abs((word['maxy'] - word['miny']) / (word['maxx'] - word['minx'] + 1e-6))
        return not (
            (word_height > size_threshold * page_ocr_data['maxy'] - page_ocr_data['miny']).median() and
            word_tilt > tilt_threshold and
            word_width > span_threshold * doc_width
        )

    filtered_data = page_ocr_data[page_ocr_data.apply(is_not_watermark, axis=1)]

    # Sort filtered data
    filtered_data = filtered_data.sort_values(['miny', 'minx'])

    # Calculate average character width
    filtered_data['char_width'] = (filtered_data['maxx'] - filtered_data['minx']) / filtered_data['word'].str.len()
    avg_char_width = filtered_data['char_width'].median()

    # Group words by lines
    line_groups = defaultdict(list)
    current_line_y = filtered_data['miny'].iloc[0]
    line_height = filtered_data['maxy'].iloc[0] - filtered_data['miny'].iloc[0]

    for _, word in filtered_data.iterrows():
        if word['miny'] - current_line_y > line_height / 2:
            num_new_empty_lines_to_be_added = int((word['miny'] - current_line_y) / line_height)
            for i in range(1, num_new_empty_lines_to_be_added):
                miny_empty = current_line_y + i*line_height/2
                line_groups[miny_empty] = []
            current_line_y = word['miny']
        line_groups[current_line_y].append(word)

    # Process each line
    result = []
    for line_y, words in sorted(line_groups.items()):
        line = ""
        current_x = 0
        for word in sorted(words, key=lambda w: w['minx']):
            spaces_before = max(0, round((word['minx'] - current_x) / avg_char_width))
            # Add a space before the word if it's not the first word on the line
            if line:
                spaces_before = max(1, spaces_before)
            line += " " * spaces_before + word['word'].replace("Ġ", "")
            current_x = word['maxx']
        result.append(line)

    return "\n".join(result)