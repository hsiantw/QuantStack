#!/bin/bash

# Script to push QuantStack to GitHub
# Run this script in the Replit Shell

echo "🚀 Setting up GitHub push for QuantStack..."

# Clean up any lock files
echo "Cleaning up git locks..."
rm -f .git/config.lock .git/index.lock

# Remove existing origin if it exists
echo "Removing existing origin..."
git remote remove origin 2>/dev/null || true

# Add the correct GitHub remote
echo "Adding GitHub remote..."
git remote add origin https://github.com/hsiantw/QuantStack.git

# Verify remote was added
echo "Verifying remote..."
git remote -v

# Check current status
echo "Current git status:"
git status

# Show current branch
echo "Current branch:"
git branch

# Attempt to push
echo "Attempting to push to GitHub..."
echo "Note: You may need to authenticate with GitHub username and token"
git push -u origin main

echo "✅ Push attempt completed!"
echo "If authentication failed, you'll need to:"
echo "1. Create a GitHub Personal Access Token"
echo "2. Use your GitHub username and token when prompted"
echo "3. Or set up Replit Secrets with GIT_URL"