import os

# To create 'data/annotations' file , this script help deletes all files in the specified directory that don't have a .tsv extension.

root_dir = '/data/annotations'

# Walk through all files and directories in the root path
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if not filename.endswith('.tsv'):
            file_path = os.path.join(dirpath, filename)
            try:
                
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
