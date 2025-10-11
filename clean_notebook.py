import nbformat
import sys

notebook_filename = 'model_comparison.ipynb'

print(f"Attempting to clean notebook: {notebook_filename}")

try:
    with open("model_comparison.ipynb", 'r', encoding='utf-8') as f:
        ntbk = nbformat.read(f, as_version=4)

    if 'widgets' in ntbk.metadata:
        print("Found problematic 'widgets' metadata. Removing it...")
        del ntbk.metadata['widgets']
        
        with open(notebook_filename, 'w', encoding='utf-8') as f:
            nbformat.write(ntbk, f)
        print("Successfully removed widget metadata. The notebook is now clean.")
    else:
        print("No 'widgets' metadata found. The notebook appears to be clean already.")

except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)