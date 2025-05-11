import json
import csv
import argparse

def json_to_csv(input_json_file, output_csv_file, max_chars=5000):
    with open(input_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt', 'target'])

        for entry in data:
            prompt = entry.get('prompt', '').strip()
            response = entry.get('response', '').strip()

            if prompt and response:
                trimmed_response = response[:max_chars]
                writer.writerow([prompt, trimmed_response])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert harmful instruction JSON to CSV with response truncation.")
    parser.add_argument('input_json', help='Path to input JSON file')
    parser.add_argument('output_csv', help='Path to output CSV file')
    args = parser.parse_args()

    json_to_csv(args.input_json, args.output_csv)