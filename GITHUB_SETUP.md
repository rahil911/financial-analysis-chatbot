# Setting Up GitHub Repository

Follow these steps to push the Financial Analysis Chatbot to GitHub so that your friend can access it:

## Creating a New GitHub Repository

1. Go to [GitHub](https://github.com) and sign in to your account.
2. Click on the "+" icon in the top right corner and select "New repository".
3. Name your repository (e.g., "financial-analysis-chatbot").
4. Add a description (optional).
5. Choose whether to make it public or private.
6. Do not initialize the repository with a README, .gitignore, or license as we already have these files.
7. Click "Create repository".

## Pushing the Code to GitHub

After creating the repository, GitHub will show instructions for pushing an existing repository. Follow these commands:

```bash
# Add the remote repository URL
git remote add origin https://github.com/YOUR_USERNAME/financial-analysis-chatbot.git

# Push the code to GitHub
git push -u origin master
```

Replace `YOUR_USERNAME` with your actual GitHub username.

## Sharing with Your Friend

Once the code is pushed to GitHub, you can share it with your friend by:

1. Giving them access to the repository if it's private:
   - Go to repository settings
   - Select "Manage access"
   - Click "Invite a collaborator" and enter their GitHub username or email

2. Or simply sharing the repository URL if it's public.

## For Your Friend: Cloning and Setting Up

Your friend should follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/financial-analysis-chatbot.git
cd financial-analysis-chatbot
```

2. Run the setup script:
```bash
# On macOS/Linux
chmod +x setup.sh
./setup.sh

# On Windows
# They might need to create a virtual environment manually and run:
# python -m venv venv
# venv\Scripts\activate
# pip install -r requirements.txt
```

3. Add OpenAI API key to .env file:
```bash
echo "export OPENAI_API_KEY='their-api-key-here'" > .env
source .env  # On Windows: use `set OPENAI_API_KEY=their-api-key-here` instead
```

4. Run the application:
```bash
python poc1/main.py --run
```

5. Open the browser at http://localhost:8501 