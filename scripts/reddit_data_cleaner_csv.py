import csv
import re
import sys

# --- FIX: Safely set the largest possible field size limit ---
# This prevents OverflowError on systems where C long is 32-bit (like Windows) 
# while still allowing very large CSV cells.
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

def filter_short_comments(input_file, output_file, min_words=10):
    input_count = 0
    output_count = 0
    total_words_saved = 0

    print(f"Filtering {input_file} (Threshold: {min_words} words)...")

    with open(input_file, 'r', encoding='utf-8', newline='') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        
        for row in reader:
            input_count += 1
            text = row.get('text', '')
            
            # Split by whitespace to count words
            words = text.split()
            word_count = len(words)
            
            if word_count >= min_words:
                writer.writerow(row)
                output_count += 1
                total_words_saved += word_count

    print("-" * 30)
    print(f"Original Records: {input_count}")
    print(f"Cleaned Records:  {output_count}")
    print(f"Records Removed:  {input_count - output_count}")
    print(f"Total Word Count: {total_words_saved}")
    print(f"Final file saved as: {output_file}")

def remove_media_and_links(input_file, output_file):
    processed_count = 0
    
    # Regex Patterns
    media_tag_pattern = re.compile(r'<[^>]+>')
    markdown_image_pattern = re.compile(r'!\[.*?\]\(.*?\)')
    url_pattern = re.compile(r'http\S+|www\S+|https\S+')

    print(f"Cleaning media and URLs from {input_file}...")

    with open(input_file, 'r', encoding='utf-8', newline='') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        
        for row in reader:
            text = row.get('text', '')

            # Apply cleaning steps
            text = media_tag_pattern.sub('', text)
            text = markdown_image_pattern.sub('', text)
            text = url_pattern.sub('', text)
            
            # Clean up double spaces
            text = " ".join(text.split())

            # Only save if the record still has substance (>= 10 words)
            if len(text.split()) >= 10:
                row['text'] = text
                writer.writerow(row)
                processed_count += 1

    print("-" * 30)
    print(f"Clean records saved: {processed_count}")
    print(f"Final file: {output_file}")

def remove_bot_comments(input_file, output_file):
    bot_string = "i am a bot, and this action was performed automatically"
    input_count = 0
    output_count = 0

    print(f"Purging bot comments from {input_file}...")

    with open(input_file, 'r', encoding='utf-8', newline='') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        
        for row in reader:
            input_count += 1
            text = row.get('text', '')
            
            # Case-insensitive bot check
            if bot_string not in text.lower():
                writer.writerow(row)
                output_count += 1

    removed = input_count - output_count
    print("-" * 30)
    print(f"Total Records Scanned: {input_count}")
    print(f"Bot Comments Removed:  {removed}")
    print(f"Clean Records Saved:   {output_count}")
    print(f"Final file: {output_file}")

def generate_report_stats(file_path):
    record_count = 0
    total_words = 0
    unique_words = set()

    with open(file_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('text', '').lower()
            
            record_count += 1
            words = text.split()
            total_words += len(words)
            unique_words.update(words)

    if record_count == 0:
        print("No data found in file.")
        return

    print(f"--- Statistics ---")
    print(f"Total Records:      {record_count}")
    print(f"Total Word Count:   {total_words}")
    print(f"Unique Words:       {len(unique_words)}")
    print(f"Avg. Words/Record:  {total_words / record_count:.2f}")

if __name__ == "__main__":
    # Example usage:
    filter_short_comments('first_100_relevant_records.csv', 'step1.csv')
    remove_media_and_links('step1.csv', 'step2.csv')
    remove_bot_comments('step2.csv', 'step3.csv')
    generate_report_stats('step3.csv')
    pass