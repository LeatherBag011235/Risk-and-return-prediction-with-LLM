import psycopg2
import json

# Database credentials
DB_NAME = "reports_db"
DB_USER = "postgres"
DB_PASSWORD = "postgres"  
DB_HOST = "localhost"
DB_PORT = "5432"   

def fetch_random_report():
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()

        # Fetch a random row
        query = "SELECT id, cik, filed_date, report_type, url, LEFT(raw_text, 50000), extracted_data, created_at, lm_orig_score FROM reports ORDER BY RANDOM() LIMIT 1;"
        cursor.execute(query)
        row = cursor.fetchone()

        if row:
            print("\n--- Random Report ---")
            print(f"ID: {row[0]}")
            print(f"CIK: {row[1]}")
            print(f"Filed Date: {row[2]}")
            print(f"Report Type: {row[3]}")
            print(f"URL: {row[4]}")
            print(f"Raw Text (First 50000 chars): {row[5]}")
            print(f"Extracted Data: {json.dumps(row[6], indent=2) if row[6] else 'None'}")
            print(f"Created At: {row[7]}")
            print(f"LM Orig Score: {row[8]}")
            print("\n---------------------")

        else:
            print("No data found.")

        cursor.close()
        conn.close()

    except Exception as e:
        print("Error:", e)

fetch_random_report()
