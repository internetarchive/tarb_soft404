import csv
import sys
from tqdm import tqdm  

def remove_duplicates_from_csv(csv_file):
    parent_urls = []
    rows_to_keep = []

    with open(csv_file, "r", newline='', encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader)
        url_index = header.index("url")

        # Wrap the loop with tqdm for progress tracking
        for line_num, row in tqdm(enumerate(reader, start=2), desc="Processing", unit="row"):
            try:
                url = row[url_index]
                parent_url = "/".join(url.split("/")[:3])
                if parent_url not in parent_urls:
                    parent_urls.append(parent_url)
                    rows_to_keep.append(row)
            except UnicodeDecodeError:
                print(f"UnicodeDecodeError: Skipped row {line_num} due to decoding issue", file=sys.stderr)

    output_file = "updated_" + csv_file
    with open(output_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows_to_keep)

    print("Duplicate parent URLs have been removed. Updated CSV file:", output_file)

# Usage
csv_file_path = "en_curlie_filter.csv"
remove_duplicates_from_csv(csv_file_path)
