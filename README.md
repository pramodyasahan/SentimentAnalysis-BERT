# Sentiment Analysis Project

This project is focused on building a sentiment analysis tool using a fine-tuned BERT model, web scraping, and model
interpretability techniques. The project leverages modern web frameworks like Next.js for the frontend and FastAPI for
the backend to create a real-time sentiment analysis application.

## Features

- **Fine-tuned BERT Model**: Utilizes the BERT model fine-tuned on the Amazon Polarity dataset for sentiment
  classification.
- **Web Scraping**: Scrapes reviews from websites like Yelp to analyze sentiment in real-time.
- **Model Interpretability**: Provides interpretability for model predictions using LIME (Local Interpretable
  Model-Agnostic Explanations).
- **Real-Time Sentiment Analysis Web App**: Built with Next.js for the frontend and FastAPI for the backend.

## Setup and Installation

### Backend

1. **Navigate to the `backend` folder:**

    ```bash
    cd backend
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the FastAPI server:**

    ```bash
    uvicorn app:app --reload
    ```

### Frontend

1. **Navigate to the `frontend` folder:**

    ```bash
    cd frontend
    ```

2. **Install dependencies:**

    ```bash
    npm install
    ```

3. **Run the Next.js development server:**

    ```bash
    npm run dev
    ```

## Usage

- **Real-Time Sentiment Analysis**: Open the frontend application in your browser and enter text to analyze the
  sentiment in real-time.
- **Model Interpretability**: Use the provided functions in `interpretability.py` to explain model predictions using
  LIME.

## Technologies Used

- **Python**: Core programming language for model training and backend.
- **FastAPI**: Backend framework to expose the sentiment analysis API.
- **Next.js**: React framework for building the frontend web application.
- **Transformers (Hugging Face)**: Used for fine-tuning the BERT model.
- **LIME**: For model interpretability.
- **Docker**: Containerization for easy deployment.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library.
- [Streamlit](https://streamlit.io/) for providing an alternative real-time interface.
- [Yelp](https://www.yelp.com/) for the source of web-scraped reviews.
