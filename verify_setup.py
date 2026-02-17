"""
Setup Verification Script

Run this script to verify that your environment is set up correctly.
"""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} - Need 3.8+")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")
    
    required = [
        'torch',
        'numpy',
        'matplotlib',
        'seaborn',
        'tqdm',
        'sklearn',
        'pandas'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    return len(missing) == 0


def check_project_structure():
    """Check if all required files exist."""
    print("\nChecking project structure...")
    
    required_files = [
        'README.md',
        'QUICKSTART.md',
        'START_HERE.md',
        'requirements.txt',
        'LICENSE',
        '.gitignore',
        'src/__init__.py',
        'src/positional_encodings.py',
        'src/model.py',
        'src/dataset.py',
        'src/train.py',
        'tests/test_positional_encoding.py',
        'scripts/compare_all.py'
    ]
    
    missing = []
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING")
            missing.append(file)
    
    return len(missing) == 0


def test_imports():
    """Test if project modules can be imported."""
    print("\nTesting module imports...")
    
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    
    modules = [
        'positional_encodings',
        'model',
        'dataset'
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except Exception as e:
            print(f"  ✗ {module} - Error: {e}")
            failed.append(module)
    
    return len(failed) == 0


def main():
    """Run all verification checks."""
    print("="*60)
    print("SETUP VERIFICATION")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Module Imports", test_imports)
    ]
    
    results = []
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:20s} {status}")
        if not result:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✅ SETUP VERIFICATION COMPLETE - All checks passed!")
        print("\nNext steps:")
        print("  1. Run tests: python tests/test_positional_encoding.py")
        print("  2. Train a model: python src/train.py --pos_encoding learned_absolute --num_epochs 10")
        print("  3. Review START_HERE.md for submission instructions")
    else:
        print("\n❌ SETUP INCOMPLETE - Some checks failed")
        print("\nPlease:")
        print("  1. Install missing dependencies: pip install -r requirements.txt")
        print("  2. Check that you're in the project directory")
        print("  3. Re-run this script")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
