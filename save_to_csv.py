import csv

def save_results_to_csv(results, filename="results.csv"):
    headers = ["URL", "Title", "Threat Level"]
    with open(filename, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    print(f"[INFO] Results saved to {filename}")
