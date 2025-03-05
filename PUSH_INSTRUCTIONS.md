# Instructions for Pushing to GitHub

Follow these steps to push the Financial Analysis Chatbot to GitHub:

1. Create a new repository on GitHub:
   - Go to https://github.com/new
   - Name it "financial-analysis-chatbot" (or your preferred name)
   - Set it to private or public as needed
   - Do NOT initialize with README, .gitignore, or license
   - Click "Create repository"

2. Copy the repository URL (https or SSH format)

3. Add the remote repository and push:
```bash
git remote add origin YOUR_REPO_URL_HERE
git push -u origin master
```

4. Share the repository with your friend by:
   - Adding them as a collaborator (if private)
   - Sharing the repository URL (if public)

## For Your Friend

Your friend should follow these steps:

1. Clone the repository
2. Run the `setup.sh` script (included in the repo)
3. The database file is already included in the repository
4. Add their OpenAI API key to the .env file
5. Run the application with `python poc1/main.py --run`

For more detailed instructions, see the `README.md` and `GITHUB_SETUP.md` files. 