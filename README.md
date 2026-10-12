# Fake News Detection Using Text Patterns

A beginner-friendly Python + Streamlit project that uses TF-IDF text features and Logistic Regression to classify a news text as **likely real** or **likely fake**, based on a small example dataset.

## Important limitation
This is an educational demonstration, not a fact-checking service. The included dataset is small and illustrative, so predictions are not reliable evidence that a story is true or false. For a serious project, use a reputable, properly licensed dataset, keep its source and labels documented, and evaluate the model on held-out data.

## Features
- Simple web interface
- Text input for a headline or article
- TF-IDF features (unigrams and bigrams)
- Logistic Regression classifier
- Prediction label and model confidence display

## Run locally
1. Install Python 3.10 or newer.
2. Download or clone this project.
3. Open a terminal in this project folder.
4. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   python -m streamlit run app.py
   ```

6. Your browser should open the local app. If it does not, open the local URL shown in the terminal.

## Dataset format
`news_dataset.csv` must contain two columns:

- `text`: headline/article text
- `label`: `real` or `fake`

Add more reliable labelled examples to improve the project. Do not use the tiny included sample to make real-world decisions.

## Upload to GitHub (no command line needed)
1. Create/sign in to a GitHub account at https://github.com/.
2. Click **+** in the top-right and choose **New repository**.
3. Name it `fake-news-detection`; choose **Public** if you want to share it.
4. Click **Create repository**.
5. On the repository page, select **Add file → Upload files**.
6. Upload `app.py`, `requirements.txt`, `news_dataset.csv`, and `README.md` from this project folder. You can upload the files individually if needed.
7. Click **Commit changes**.

## Optional: publish a live web app
GitHub stores the source code but does not automatically run this Streamlit app. To publish it, use Streamlit Community Cloud:
1. Visit https://share.streamlit.io/ and sign in with GitHub.
2. Choose **Create app** and select your repository.
3. Set the main file path to `app.py`.
4. Deploy the app and wait for the build to finish.

Availability and deployment settings may change. Never upload passwords, API keys, or private data to a public repository.
