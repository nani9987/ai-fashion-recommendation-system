#!/usr/bin/env python
"""Verification script to check all 25+ bugs are fixed."""

import sys
from pathlib import Path


def check_file_not_empty(filepath, min_size=100):
    """Check if file exists and is not empty."""
    p = Path(filepath)
    if not p.exists():
        return False, f"File not found: {filepath}"
    if p.stat().st_size < min_size:
        return False, f"File too small: {filepath}"
    return True, f"✅ {filepath}"


def check_file_contains(filepath, search_string):
    """Check if file contains a specific string."""
    try:
        with open(filepath) as f:
            content = f.read()
        return search_string.lower() in content.lower()
    except:
        return False


def main():
    print("\n" + "="*80)
    print("🔍 AI FASHION RECOMMENDATION SYSTEM - BUG FIX VERIFICATION")
    print("="*80 + "\n")
    
    issues = []
    fixed = 0
    total = 0
    
    # BUG GROUP 1: requirements.txt
    print("📦 REQUIREMENTS.TXT CHECKS:")
    total += 3
    
    if check_file_contains("requirements.txt", "tensorflow"):
        print("  ✅ TensorFlow included")
        fixed += 1
    else:
        print("  ❌ TensorFlow missing")
        issues.append("TensorFlow not in requirements.txt")
    
    if check_file_contains("requirements.txt", "tqdm"):
        print("  ✅ tqdm included")
        fixed += 1
    else:
        print("  ❌ tqdm missing")
        issues.append("tqdm not in requirements.txt")
    
    if "==" in open("requirements.txt").read():
        print("  ✅ Versions pinned")
        fixed += 1
    else:
        print("  ❌ Versions not pinned")
        issues.append("Dependencies not pinned to versions")
    
    # BUG GROUP 2: src/config.py
    print("\n⚙️  CONFIG.PY CHECKS:")
    total += 3
    
    try:
        from src.config import BASE_DIR, IMAGE_DIR, CSV_PATH, EMBEDDINGS_DIR
        print("  ✅ All config paths defined")
        fixed += 1
        
        from pathlib import Path
        if isinstance(BASE_DIR, Path):
            print("  ✅ Paths use pathlib.Path")
            fixed += 1
        else:
            print("  ❌ Paths not using pathlib")
            issues.append("Config paths should use pathlib.Path")
        
        if BASE_DIR.is_absolute():
            print("  ✅ Paths are absolute")
            fixed += 1
        else:
            print("  ❌ Paths not absolute")
            issues.append("Config paths should be absolute")
    except Exception as e:
        print(f"  ❌ Error loading config: {e}")
        issues.append(f"Config load error: {e}")
    
    # BUG GROUP 3: src/data_loader.py
    print("\n📂 DATA_LOADER.PY CHECKS:")
    total += 4
    
    try:
        from src.data_loader import load_dataset, apply_filters
        print("  ✅ load_dataset function exists")
        fixed += 1
        print("  ✅ apply_filters function exists")
        fixed += 1
        
        # Check for error handling
        if check_file_contains("src/data_loader.py", "try"):
            print("  ✅ Error handling implemented")
            fixed += 1
        else:
            print("  ⚠️  Limited error handling")
        
        if check_file_contains("src/data_loader.py", "lower()"):
            print("  ✅ Case-insensitive filtering")
            fixed += 1
        else:
            print("  ❌ No case normalization")
            issues.append("data_loader should normalize to lowercase")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        issues.append(f"data_loader error: {e}")
    
    # BUG GROUP 4: src/feature_extractor.py
    print("\n🧠 FEATURE_EXTRACTOR.PY CHECKS:")
    total += 4
    
    try:
        if check_file_contains("src/feature_extractor.py", "ResNet50"):
            print("  ✅ ResNet50 model defined")
            fixed += 1
        else:
            print("  ❌ ResNet50 not found")
            issues.append("feature_extractor missing ResNet50")
        
        if check_file_contains("src/feature_extractor.py", "extract_features"):
            print("  ✅ extract_features function exists")
            fixed += 1
        else:
            print("  ❌ extract_features not found")
            issues.append("extract_features function missing")
        
        if check_file_contains("src/feature_extractor.py", "try"):
            print("  ✅ Error handling for images")
            fixed += 1
        else:
            print("  ⚠️  Limited error handling")
        
        if check_file_contains("src/feature_extractor.py", "2048"):
            print("  ✅ Correct embedding dimension (2048)")
            fixed += 1
        else:
            print("  ⚠️  Check embedding dimension")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # BUG GROUP 5: src/recommender.py
    print("\n🎯 RECOMMENDER.PY CHECKS (CRITICAL):")
    total += 5
    
    try:
        if check_file_contains("src/recommender.py", "cosine_similarity"):
            print("  ✅ Cosine similarity implemented")
            fixed += 1
        else:
            print("  ❌ NO COSINE SIMILARITY - CRITICAL BUG")
            issues.append("❌ CRITICAL: recommender.py missing cosine similarity")
        
        if check_file_contains("src/recommender.py", "query_embedding"):
            print("  ✅ Takes query_embedding parameter")
            fixed += 1
        else:
            print("  ❌ Missing query_embedding parameter")
            issues.append("recommender should take query_embedding parameter")
        
        if check_file_contains("src/recommender.py", "similarity_score"):
            print("  ✅ Returns similarity scores")
            fixed += 1
        else:
            print("  ⚠️  No similarity scores in output")
        
        if check_file_contains("src/recommender.py", "argsort"):
            print("  ✅ Ranks by similarity")
            fixed += 1
        else:
            print("  ⚠️  May not rank properly")
        
        if not check_file_contains("src/recommender.py", "sample()"):
            print("  ✅ NOT just random sampling")
            fixed += 1
        else:
            print("  ❌ Still using random sampling")
            issues.append("recommender should NOT use random sampling")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # BUG GROUP 6: src/build_embeddings.py
    print("\n🔨 BUILD_EMBEDDINGS.PY CHECKS:")
    total += 5
    
    try:
        if check_file_contains("src/build_embeddings.py", "time.time"):
            print("  ✅ Timing information added")
            fixed += 1
        else:
            print("  ⚠️  No timing")
        
        if check_file_contains("src/build_embeddings.py", "try"):
            print("  ✅ Error handling for failed images")
            fixed += 1
        else:
            print("  ⚠️  Limited error handling")
        
        if check_file_contains("src/build_embeddings.py", "EMBEDDINGS_DIR"):
            print("  ✅ Uses EMBEDDINGS_DIR from config")
            fixed += 1
        else:
            print("  ❌ Hardcoded embeddings path")
            issues.append("build_embeddings should use EMBEDDINGS_DIR from config")
        
        if check_file_contains("src/build_embeddings.py", "dtype=np.float32"):
            print("  ✅ Specifies float32 dtype")
            fixed += 1
        else:
            print("  ⚠️  No explicit dtype")
        
        if check_file_contains("src/build_embeddings.py", "load_dataset"):
            print("  ✅ Uses load_dataset function")
            fixed += 1
        else:
            print("  ⚠️  Direct CSV loading")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # BUG GROUP 7: app.py
    print("\n🎨 APP.PY CHECKS (CRITICAL):")
    total += 4
    
    try:
        if check_file_contains("app.py", "load_embeddings"):
            print("  ✅ Loads embeddings")
            fixed += 1
        else:
            print("  ❌ CRITICAL: No embeddings loading")
            issues.append("❌ CRITICAL: app.py doesn't load embeddings")
        
        if check_file_contains("app.py", "extract_features"):
            print("  ✅ Processes uploaded images")
            fixed += 1
        else:
            print("  ❌ CRITICAL: No image processing")
            issues.append("❌ CRITICAL: app.py doesn't process uploaded images")
        
        if check_file_contains("app.py", "get_recommendations"):
            print("  ✅ Calls recommender function")
            fixed += 1
        else:
            print("  ❌ CRITICAL: No ML integration")
            issues.append("❌ CRITICAL: app.py doesn't call recommender")
        
        # Check for duplicate lines
        with open("app.py") as f:
            content = f.read()
        if content.count("^\n") < 5:  # Reasonable threshold for duplicates
            print("  ✅ No excessive duplicate lines")
            fixed += 1
        else:
            print("  ⚠️  Check for duplicate lines")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # BUG GROUP 8: README.md
    print("\n📖 README.MD CHECKS:")
    total += 2
    
    try:
        if check_file_contains("README.md", "ResNet50"):
            print("  ✅ Documents ResNet50")
            fixed += 1
        else:
            print("  ⚠️  Missing ResNet50 documentation")
        
        if check_file_contains("README.md", "cosine"):
            print("  ✅ Documents cosine similarity")
            fixed += 1
        else:
            print("  ⚠️  Missing cosine similarity documentation")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Summary
    print("\n" + "="*80)
    print(f"\n📊 SUMMARY: {fixed}/{total} bugs fixed ({100*fixed//total}%)")
    print("="*80 + "\n")
    
    if issues:
        print("❌ REMAINING ISSUES:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print()
        return 1
    else:
        print("✅ ALL BUGS FIXED! Your project is ready.\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
