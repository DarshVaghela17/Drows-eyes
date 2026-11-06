"""
Database & Event Logging Service
Persists real-time driver drowsiness telemetry events to SQLite (drowsiness_logs.db)
with automatic CSV fallback and query utilities for audit reporting.
"""

import csv
import os
import sqlite3
from datetime import datetime

class DrowsinessDatabase:
    def __init__(self, db_path='drowsiness_logs.db', csv_path='drowsiness_events.csv'):
        self.db_path = db_path
        self.csv_path = csv_path
        self._init_sqlite()
        self._init_csv()

    def _init_sqlite(self):
        """Initializes SQLite table schemas"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                driver_state TEXT NOT NULL,
                ear REAL NOT NULL,
                mar REAL NOT NULL,
                head_pitch REAL NOT NULL,
                blink_count INTEGER NOT NULL,
                perclos REAL NOT NULL,
                confidence REAL NOT NULL,
                alarm_triggered INTEGER NOT NULL,
                notes TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def _init_csv(self):
        """Creates CSV header if file does not exist"""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'driver_state', 'ear', 'mar',
                    'head_pitch', 'blink_count', 'perclos',
                    'confidence', 'alarm_triggered', 'notes'
                ])

    def log_event(self, state, ear, mar, head_pitch, blink_count, perclos, confidence, alarm_triggered, notes=''):
        """Inserts detection record into SQLite and appends to CSV"""
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # 1. SQLite Write
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO detections (
                    timestamp, driver_state, ear, mar, head_pitch,
                    blink_count, perclos, confidence, alarm_triggered, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (now_str, state, ear, mar, head_pitch, blink_count, perclos, confidence, int(alarm_triggered), notes))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[DB Warning] SQLite write error: {e}")

        # 2. CSV Append
        try:
            with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    now_str, state, f"{ear:.3f}", f"{mar:.3f}",
                    f"{head_pitch:.1f}", blink_count, f"{perclos:.1f}",
                    f"{confidence:.2f}", int(alarm_triggered), notes
                ])
        except Exception as e:
            print(f"[DB Warning] CSV append error: {e}")

    def fetch_recent_logs(self, limit=50):
        """Fetches last N logs as a list of dictionaries for dashboard display"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM detections ORDER BY id DESC LIMIT ?', (limit,))
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_summary_statistics(self):
        """Returns aggregate metrics: total scans, severe drowsiness count, average EAR"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*), AVG(ear), AVG(mar) FROM detections')
        total, avg_ear, avg_mar = cursor.fetchone()

        cursor.execute("SELECT COUNT(*) FROM detections WHERE driver_state = 'Severe Drowsiness'")
        severe_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM detections WHERE alarm_triggered = 1")
        alarms_count = cursor.fetchone()[0]

        conn.close()
        return {
            'total_events': total or 0,
            'avg_ear': round(avg_ear or 0.0, 3),
            'avg_mar': round(avg_mar or 0.0, 3),
            'severe_events': severe_count or 0,
            'alarms_triggered': alarms_count or 0
        }
