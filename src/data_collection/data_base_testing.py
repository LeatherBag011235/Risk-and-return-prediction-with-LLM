import psycopg2

# Database credentials
DB_NAME = "reports_db"
DB_USER = "postgres"
DB_PASSWORD = "postgres"  # Replace with your actual password
DB_HOST = "localhost"
DB_PORT = "5432"

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

    # 1️⃣ Insert a sample company (if not exists)
    cursor.execute("""
        INSERT INTO companies (cik, name, ticker, sector, industry)
        VALUES ('0001234', 'Test Corp', 'TSTC', 'Technology', 'Software')
        ON CONFLICT (cik) DO NOTHING;
    """)

    # 2️⃣ Insert a sample SEC report
    sample_text = "This is a sample SEC 10-Q report for Test Corp."
    cursor.execute("""
        INSERT INTO reports (cik, filing_date, report_type, url, raw_text)
        VALUES ('0001234', '2024-03-01', '10-Q', 'https://sec.gov/test_report', %s)
        RETURNING id;
    """, (sample_text,))

    report_id = cursor.fetchone()[0]
    conn.commit()

    print(f"✅ Inserted sample report with ID {report_id}")

    # 3️⃣ Retrieve and print the inserted data
    cursor.execute("SELECT cik, filing_date, report_type, raw_text FROM reports WHERE id = %s", (report_id,))
    report = cursor.fetchone()

    print("\n📄 Retrieved Report:")
    print(f"CIK: {report[0]}")
    print(f"Filing Date: {report[1]}")
    print(f"Report Type: {report[2]}")
    print(f"Text: {report[3]}")

    # Close connection
    cursor.close()
    conn.close()

except Exception as e:
    print(f"❌ Error: {e}")
