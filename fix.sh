# 1. Confirm branches
git branch

# 2. Switch to the intended branch
git checkout feature-great-lakes

# 3. Merge or move your current work
# (if changes are still in working directory, they’ll carry over automatically)
# if you already committed in the wrong branch:
git cherry-pick <commit_hash_from_feature-initial-build>

# 4. Push to remote
git push origin feature-great-lakes

