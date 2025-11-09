# 1. See which file has the conflict
git status

# 2. Open and fix it manually (remove <<<<<<<, =======, >>>>>>> lines)
nano src/yolo_v8/yolo_v8_0.py

# 3. Mark as resolved
git add src/yolo_v8/yolo_v8_0.py

# 4. Commit merge
git commit -m "Resolved merge conflict"

# 5. Push to remote
git push origin feature-initial-build

