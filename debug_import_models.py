try:
    from app.db import models
    print("Import successful!")
except Exception as e:
    import traceback
    print("Import failed:\n")
    traceback.print_exc()
