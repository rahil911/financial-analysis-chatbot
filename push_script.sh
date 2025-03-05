#!/bin/bash

# This script pushes to GitHub and then deletes itself for security
echo "Pushing code to GitHub..."

# Force push all branches and tags
git push -u origin master

# Let the user know it's done
echo "Code pushed to GitHub successfully!"
echo "Your repository should now be available at: https://github.com/rahil911/financial-analysis-chatbot"
echo ""
echo "To share with your friend, send them the repository URL or add them as a collaborator." 