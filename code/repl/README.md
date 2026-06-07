<!-- 
TAG_NAME="release-2026-06-05"
git tag -d "${TAG_NAME}"
git push origin --delete "${TAG_NAME}"
git fetch --prune --prune-tags origin

TAG_NAME="release-$(date +%Y-%m-%d)"
git tag -a "${TAG_NAME}" -m "Release ${TAG_NAME}"
git push origin "${TAG_NAME}" 
-->