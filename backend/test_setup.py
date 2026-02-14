"""
Verify all libraries are installed correctly
"""
import sys

print("=" * 60)
print("  TIMETABLE AI - ENVIRONMENT CHECK")
print("=" * 60)
print(f"\nPython Version: {sys.version}")
print(f"Python Path: {sys.executable}\n")

libraries = [
    ("FastAPI", "fastapi"),
    ("Uvicorn", "uvicorn"),
    ("DEAP (Genetic Algorithms)", "deap"),
    ("NumPy", "numpy"),
    ("Pandas", "pandas"),
    ("Scikit-learn", "sklearn"),
    ("PyTorch", "torch"),
    ("SQLAlchemy", "sqlalchemy"),
    ("Pydantic", "pydantic"),
]

print("Checking installed libraries...\n")
all_good = True

for name, module in libraries:
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", "unknown")
        print(f"✅ {name:30} v{version}")
    except ImportError:
        print(f"❌ {name:30} MISSING!")
        all_good = False

print("\n" + "=" * 60)
if all_good:
    print("🎉 All libraries installed successfully!")
    print("✅ Environment is ready for development")
else:
    print("⚠️  Some libraries are missing. Please reinstall.")
print("=" * 60)
