import sys
import traceback

try:
    from app import app
    print("Successfully imported app from app.py")
    print(f"App routes: {app.url_map}")
except Exception as e:
    print(f"Error importing app: {str(e)}")
    traceback.print_exc()
    
try:
    from main import app
    print("\nSuccessfully imported app from main.py")
    print(f"App routes: {app.url_map}")
except Exception as e:
    print(f"\nError importing app from main.py: {str(e)}")
    traceback.print_exc()