# GitHub Setup Instructions

## Important Security Note
I noticed you shared your GitHub password. For security reasons, you should:
1. Change your GitHub password immediately after this session
2. Never share your password in plain text

## Steps to Create a GitHub Repository

1. Go to https://github.com/new in your browser
2. Log in with your GitHub account
3. Set up the repository:
   - Name: financial-analysis-chatbot
   - Description: An intelligent financial analysis chatbot powered by OpenAI's GPT models
   - Visibility: Private or Public as you prefer
   - Do NOT initialize with README, .gitignore, or license
   - Click "Create repository"

## After Creating the Repository

After creating the repository, come back to this terminal and run:

```bash
./push_script.sh
```

When prompted for authentication, use your GitHub username and a Personal Access Token instead of your password.

## How to Create a Personal Access Token (Safer than using your password)

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Generate new token
2. Give it a name like "Financial Analysis Chatbot"
3. Set expiration as needed (e.g., 7 days)
4. Select scopes: at minimum check "repo" (full control of repositories)
5. Click "Generate token"
6. Copy the token immediately (it will only be shown once)

## Sharing with Your Friend

Once pushed, you can share the repository with your friend by:
1. Adding them as a collaborator (if private)
2. Simply sharing the URL (if public): https://github.com/rahil911/financial-analysis-chatbot

Your friend can then follow the instructions in README.md and GITHUB_SETUP.md to set up and run the chatbot. 