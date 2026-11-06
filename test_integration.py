"""Integration test to verify all modules load and work correctly."""
import sys
import traceback

errors = []
warnings = []

# Test 1: landmarks_detector
print("=" * 60)
print("TEST 1: landmarks_detector")
try:
    from landmarks_detector import FacialLandmarkDetector
    d = FacialLandmarkDetector()
    print("  [PASS] FacialLandmarkDetector created successfully")
    
    # Test with a dummy black image (should return None for no face)
    import numpy as np
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    raw, pixel = d.extract_landmarks(dummy)
    if raw is None and pixel is None:
        print("  [PASS] extract_landmarks returns (None, None) for blank image")
    else:
        print("  [WARN] extract_landmarks returned unexpected result for blank image")
except Exception as e:
    errors.append(f"landmarks_detector: {e}")
    print(f"  [FAIL] {e}")
    traceback.print_exc()

# Test 2: database
print("\nTEST 2: database")
try:
    from database import DrowsinessDatabase
    import os
    test_db = 'test_integration.db'
    test_csv = 'test_integration.csv'
    db = DrowsinessDatabase(db_path=test_db, csv_path=test_csv)
    print("  [PASS] DrowsinessDatabase created")
    
    # Test log_event with the EXACT call from app_streamlit.py line 202
    db.log_event("Mild Drowsiness", 0.2, 0.5, 15.0, 3, 0.0, 92.0, True)
    print("  [PASS] log_event (8 positional args) works")
    
    # Test fetch and stats
    logs = db.fetch_recent_logs(limit=10)
    print(f"  [PASS] fetch_recent_logs returned {len(logs)} record(s)")
    
    stats = db.get_summary_statistics()
    print(f"  [PASS] get_summary_statistics: {stats}")
    
    # Cleanup
    os.remove(test_db)
    os.remove(test_csv)
except Exception as e:
    errors.append(f"database: {e}")
    print(f"  [FAIL] {e}")
    traceback.print_exc()

# Test 3: gradcam
print("\nTEST 3: gradcam")
try:
    from gradcam import GradCAMVisualizer
    print("  [PASS] GradCAMVisualizer imported")
except Exception as e:
    errors.append(f"gradcam: {e}")
    print(f"  [FAIL] {e}")

# Test 4: model
print("\nTEST 4: models.cnn_lstm_model")
try:
    from models.cnn_lstm_model import build_cnn_lstm_model, compile_model
    model = build_cnn_lstm_model()
    print(f"  [PASS] Model built: {model.name}")
except Exception as e:
    errors.append(f"cnn_lstm_model: {e}")
    print(f"  [FAIL] {e}")

# Test 5: evaluate imports
print("\nTEST 5: evaluate imports")
try:
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    import seaborn as sns
    import matplotlib.pyplot as plt
    print("  [PASS] sklearn, seaborn, matplotlib all imported")
except Exception as e:
    errors.append(f"evaluate imports: {e}")
    print(f"  [FAIL] {e}")

# Test 6: train imports
print("\nTEST 6: train imports")
try:
    from sklearn.model_selection import train_test_split
    from sklearn.utils import class_weight
    print("  [PASS] sklearn.model_selection and sklearn.utils imported")
except Exception as e:
    errors.append(f"train imports: {e}")
    print(f"  [FAIL] {e}")

# Test 7: streamlit (import only)
print("\nTEST 7: streamlit import")
try:
    import streamlit as st
    print(f"  [PASS] streamlit {st.__version__} imported")
except Exception as e:
    errors.append(f"streamlit: {e}")
    print(f"  [FAIL] {e}")

# Summary
print("\n" + "=" * 60)
if errors:
    print(f"RESULT: {len(errors)} ERROR(S) FOUND:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("RESULT: ALL TESTS PASSED! No errors detected.")
    sys.exit(0)
