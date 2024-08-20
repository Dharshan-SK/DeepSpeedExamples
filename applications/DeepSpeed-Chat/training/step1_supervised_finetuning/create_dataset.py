from utils import get_ocr_data_for_doc, get_processed_page_ocr_data, ocr_to_aligned_text_no_watermark
from tqdm import tqdm
import os
import json
import pandas as pd

# test data
# s3://javis-ai-parser-dev/ai_parser_testing/pharma_short_test_may_2024_20240501_20240531/outputs/20240724_2/General_Trade_IND_Purchase_Order_PHARMA_PO/

# train data
# march set

PROMPT = 'You are an expert AI in document parsing. Your task is to process pharmaceutical Purchase Orders (POs), which contain medical products and their ordered quantities. These POs may contain quantities of the form of [X+Y], where X is the ordered quantity and Y is the free quantity. Please extract the ordered items, and their respective quantities from the provided PO and output them in a CSV format with the columns: ["Item_Desc", "Qty", "Free_Qty"]. Ensure all rows are included and there are no duplicates.'

def process_ocr(doc_id):
    page_number = 0
    ocr_data = get_ocr_data_for_doc(doc_id)
    page_ocr_data = get_processed_page_ocr_data(ocr_data, page_number)
    ocr_text = ocr_to_aligned_text_no_watermark(page_ocr_data)
    ocr_text.replace("_", " ")
    return ocr_text

doc_ids = []



def build_prompt(i, ocr_text, output):
    ele = {}
    ele["prompt"]=PROMPT+'\n\n'+ocr_text
    ele["chosen"] = output
    
    return ele

json_path = "/home/ubuntu/data"
i=0
json_data = []
for file in tqdm(sorted(os.listdir(json_path))):
    
    if i>500:
        break
    try:
        data = json.load(open(f'{json_path}/{file}', 'r'))
        val_fail = data["output"]["validation_fail"]
        if val_fail:
            continue

        parsed_table = data["output"]["parsed_data"][0]["table"]
        if len(parsed_table)>15:
            continue
        table = pd.DataFrame(parsed_table)[["Item_Desc", "Qty", "Free_Qty"]]
        csv_string = table.to_csv(index=False)
        doc_id = file.split(".")[0]
        ocr_text = process_ocr(doc_id)
        # s3_image_file = f'training_data/information_extraction/LayoutLMv3/pharma_combined_20231201_20240430_triplehead/images/{doc_id}_0.png'
        # local_file_name = f'/home/ubuntu/dataset/images/{doc_id}.png'
    
        ele = build_prompt(i, ocr_text, csv_string)
        json_data.append(ele)
    
        i+=1
    except:
        continue

json.dump(json_data, open("/home/ubuntu/dataset/json_data.json", "w"))