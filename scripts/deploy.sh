#!/bin/bash

# Deploy script - pushes to both GitHub and Hugging Face
# Usage: ./scripts/deploy.sh [commit_message]

set -e  # Exit on any error

echo "🚀 Deploying to GitHub and Hugging Face..."

# Get commit message from argument or use default
if [ -n "$1" ]; then
    commit_msg="$1"
else
    commit_msg="feat: Deploy latest changes"
fi

# Check if there are changes to commit
if git diff --quiet && git diff --cached --quiet; then
    echo "ℹ️  No changes to commit"
else
    echo "📝 Committing changes: $commit_msg"
    git add -A
    git commit -m "$commit_msg"
fi

# Push to GitHub
echo "📤 Pushing to GitHub..."
if git push origin main; then
    echo "✅ Successfully pushed to GitHub"
else
    echo "❌ Failed to push to GitHub"
    exit 1
fi

# Push to Hugging Face
echo "📤 Pushing to Hugging Face..."
if git push hf main; then
    echo "✅ Successfully deployed to Hugging Face!"
    echo "🔗 Check deployment at: https://huggingface.co/spaces/colin730/SummarizerApp"
    echo "⏱️  Build time: ~5-10 minutes"
    echo ""
    echo "🎯 Monitor deployment with:"
    echo "   curl -s https://colin730-summarizerapp.hf.space/health"
else
    echo "❌ Failed to push to Hugging Face"
    exit 1
fi

echo ""
echo "🎉 Deployment complete! Both GitHub and Hugging Face are updated."



