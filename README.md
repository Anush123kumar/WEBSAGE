# WEBSAGE
Websage is an AI-based project that combines web scraping, vector embeddings, and generative AI to extract and analyze web content. The tool enables users to scrape data from any webpage, store it in a vector database (MILVUS), and query the stored data with questions, generating insightful answers using GEMINI.

Usage
The Websage-Bot works by first scraping text data from a specified URL, converting the text into vector embeddings using the SentenceTransformer model, and storing these embeddings in Milvus. When a user submits a query, the bot retrieves the most relevant data from the Milvus database and uses the Gemini API to generate a contextually accurate response. All interactions, including loading data and querying, are handled through API endpoints exposed via FastAPI and can be tested using the built-in Swagger UI.

Modules
Web Scraping: The bot scrapes textual data from a given webpage using BeautifulSoup. This module extracts all paragraph elements and returns clean, structured text for further processing.

Embedding Generation: Once the text is scraped, it is converted into vector embeddings using the SentenceTransformer model from the sentence-transformers library. These embeddings are stored in the Milvus vector database for efficient similarity search.

Milvus Vector Database: The Milvus database stores the embeddings and enables fast retrieval of similar data using vector similarity search. Queries to Milvus return the most relevant embeddings for a given user input.

Generative AI (Gemini): After retrieving relevant context from Milvus, the bot uses the Gemini model API to generate a detailed answer to the user’s query. The Gemini model is designed to provide coherent and contextually appropriate responses based on the input data.

FastAPI Interface: The Websage-Bot’s API is built using FastAPI, and it leverages Swagger UI to document and interact with the API endpoints. This allows users to easily load data into the vector database and query it via simple HTTP requests.

API Endpoints
The FastAPI interface exposes the following key endpoints:

/load: This POST endpoint accepts a URL, scrapes the webpage for content, processes it into embeddings, and stores the results in the Milvus vector database.
/query: This POST endpoint accepts a user query, retrieves relevant embeddings from the database, and generates a response using the Gemini API.
Both endpoints can be tested and interacted with through the Swagger UI provided by FastAPI.

Technologies
Websage-Bot is built using several core technologies:

Python: The primary programming language.
BeautifulSoup: Used for scraping text from web pages.
Sentence Transformers: Provides the embeddings for the scraped text.
Milvus: Handles the storage and similarity search of vector embeddings.
Gemini API: Generates responses based on the retrieved context.
FastAPI: Exposes the API and integrates with Swagger UI interaction.
