"""
Automated Test Suite for ANPR System.
Verifies syntax, imports, file structure, history tracking, preprocessing, OCR cleaning, and detector fallbacks.
"""

import sys
import py_compile
import numpy as np
import cv2
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

def run_tests():
    print("=" * 60)
    print("  ANPR SYSTEM - AUTOMATED VERIFICATION TEST SUITE")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = 0

    # -------------------------------------------------------------
    # Test 1: Python Syntax Compilation
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[TEST 1] Verifying Python Syntax for all project files...")
    py_files = [
        BASE_DIR / "app.py",
        BASE_DIR / "config.py",
        BASE_DIR / "train.py",
        BASE_DIR / "src" / "__init__.py",
        BASE_DIR / "src" / "detector.py",
        BASE_DIR / "src" / "preprocessing.py",
        BASE_DIR / "src" / "ocr.py",
        BASE_DIR / "src" / "history.py",
        BASE_DIR / "src" / "utils.py",
    ]
    
    syntax_ok = True
    for file_path in py_files:
        try:
            py_compile.compile(str(file_path), doraise=True)
            print(f"  [PASS] {file_path.relative_to(BASE_DIR)}: Syntax valid")
        except py_compile.PyCompileError as e:
            print(f"  [FAIL] {file_path.relative_to(BASE_DIR)}: Syntax Error! {e}")
            syntax_ok = False
            
    if syntax_ok:
        passed_tests += 1

    # -------------------------------------------------------------
    # Test 2: Directory Structure Verification
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[TEST 2] Verifying Project Directory Structure...")
    import config
    config.ensure_directories()

    required_dirs = [
        BASE_DIR / "models",
        BASE_DIR / "data",
        BASE_DIR / "data" / "images",
        BASE_DIR / "data" / "labels",
        BASE_DIR / "test_images",
        BASE_DIR / "outputs",
        BASE_DIR / "history",
        BASE_DIR / "src"
    ]
    dirs_ok = True
    for folder in required_dirs:
        if folder.exists() and folder.is_dir():
            print(f"  [PASS] Directory exists: {folder.relative_to(BASE_DIR)}")
        else:
            print(f"  [FAIL] Missing directory: {folder.relative_to(BASE_DIR)}")
            dirs_ok = False
    if dirs_ok:
        passed_tests += 1

    # -------------------------------------------------------------
    # Test 3: History CSV Operations
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[TEST 3] Testing History Tracker (CSV Creation, Append, Load, Clear)...")
    try:
        from src.history import save_detection, load_history, clear_history, get_dashboard_stats
        
        # Test save
        saved = save_detection("MH12AB1234", 0.95, 0.88, source="Test")
        print(f"  [PASS] Save detection log: {saved}")
        
        # Test load
        df = load_history()
        print(f"  [PASS] Loaded history dataframe with {len(df)} row(s)")
        assert not df.empty, "DataFrame should not be empty after save"
        
        # Test dashboard stats
        stats = get_dashboard_stats()
        print(f"  [PASS] Dashboard Stats: Total={stats['total_detections']}, Unique={stats['unique_plates']}, Latest='{stats['latest_plate']}'")
        assert stats["total_detections"] >= 1
        
        # Test clear
        cleared = clear_history()
        print(f"  [PASS] Clear history CSV: {cleared}")
        df_cleared = load_history()
        assert df_cleared.empty, "DataFrame should be empty after clear"
        
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] History test failed: {e}")

    # -------------------------------------------------------------
    # Test 4: Preprocessing Pipeline & Stage Outputs
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[TEST 4] Testing OpenCV Preprocessing Pipeline...")
    try:
        from src.preprocessing import preprocess_plate_for_ocr, get_preprocessing_stages, validate_crop
        
        # Synthetic dummy plate crop (100x300 BGR image)
        dummy_crop = np.zeros((100, 300, 3), dtype=np.uint8)
        cv2.putText(dummy_crop, "MH12AB1234", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        assert validate_crop(dummy_crop), "Crop validation failed"
        
        processed_img = preprocess_plate_for_ocr(dummy_crop)
        print(f"  [PASS] Preprocessing output shape: {processed_img.shape}, dtype: {processed_img.dtype}")
        assert len(processed_img.shape) == 2, "Preprocessed image must be 2D grayscale/binary"

        stages = get_preprocessing_stages(dummy_crop)
        print(f"  [PASS] Extracted {len(stages)} intermediate stages for Streamlit preview: {list(stages.keys())}")
        assert len(stages) == 5
        
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Preprocessing test failed: {e}")

    # -------------------------------------------------------------
    # Test 5: OCR Character Cleaning & Format Validator
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[TEST 5] Testing OCR Text Cleaning & Indian Format Validation...")
    try:
        from src.ocr import clean_ocr_text, validate_indian_plate_format
        
        raw_ocr_output = " mh 12-ab 1234! "
        cleaned = clean_ocr_text(raw_ocr_output)
        print(f"  [PASS] Raw: '{raw_ocr_output}' -> Cleaned: '{cleaned}'")
        assert cleaned == "MH12AB1234", f"Expected 'MH12AB1234', got '{cleaned}'"

        fmt_info = validate_indian_plate_format(cleaned)
        print(f"  [PASS] Indian Registration Format Check: {fmt_info}")
        assert fmt_info["is_valid_format"] is True
        
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] OCR cleaning test failed: {e}")

    # -------------------------------------------------------------
    # Test 6: Detector Fallback (Missing best.pt handling)
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[TEST 6] Testing Detector Fallback (Missing model handling)...")
    try:
        from src.detector import NumberPlateDetector
        
        detector = NumberPlateDetector()
        print(f"  [PASS] Detector initialized. Demo mode: {detector.is_demo_mode}")
        
        # Test detection on sample image
        test_img = np.zeros((400, 600, 3), dtype=np.uint8)
        result = detector.detect(test_img)
        
        print(f"  [PASS] Detection result success: {result['success']}, Detections count: {len(result['detections'])}")
        assert result["success"] is True
        assert len(result["detections"]) > 0
        
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] Detector test failed: {e}")

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  TEST SUITE RESULTS: {passed_tests} / {total_tests} PASSED")
    print("=" * 60)

    if passed_tests == total_tests:
        print("\n[SUCCESS] ALL AUDIT CHECKS PASSED SUCCESSFULLY!")
        return 0
    else:
        print("\n[WARNING] SOME AUDIT CHECKS FAILED.")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())
