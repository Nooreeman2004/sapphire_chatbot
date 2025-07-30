import os

print("\nDirectories in current path:")
for item in os.listdir():
    if os.path.isdir(item):
        print("-", item)
