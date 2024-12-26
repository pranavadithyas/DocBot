# Hackathon
# Doc Bot

## Problem Statement
Design and implement a chatbot system capable of ingesting and interpreting
uploaded documents (e.g., PDFs) to provide accurate, fact-based responses
quickly and reliably. The chatbot should utilize LLM APIs and other retrieval
techniques.

### Deliverables:
1. Responsive REST APIs connected with Simple UI

### Good to have:
- Support bulk upload and processing.
- Security by design.
- Improved User experience using UI and streaming APIs.
- Employ effective techniques (e.g., prompt engineering, context-verification, or grounding) to prevent “hallucinations” by verifying that all responses directly reference the source material.

---

## Solution Abstract
This project implements a Retrieval-Augmented Generation (RAG) model to provide accurate and contextually relevant answers to user queries. By combining document retrieval and large language model (LLM) capabilities, the assistant ensures reliable and precise responses. The docbot uses:

- **Document Loading**: Load and preprocess data from uploaded directories or web-based sources.
- **Text Splitting**: Efficiently chunk large documents for better embedding performance.
- **Vector Store**: Create and store embeddings generated using Sentence Transformers in a vector database for fast similarity search.
- **Retriever**: Retrieve the most relevant documents based on the user’s query.
- **LLM (gemini-pro)**: Generate conversational, user-friendly answers.
- **Flask Backend**: Simple and responsive API-based interaction for streamlined functionality.

A video demonstration is included below to showcase the application's functionality and workflow.

---

## Features
1. **Document Retrieval**: Loads documents from directories or the web, processes them into searchable chunks.
2. **Embeddings Generation**: Use Sentence Transformers to create semantic embeddings of document content.
3. **RAG Pipeline**: Combine document retrieval and generation via Google Generative AI API to produce fact-based and reliable responses.
4. **Flask Backend**: A RESTful backend for real-time query handling and response generation.
5. **Session Management**: Caches vector stores and history to enhance performance.

---

## Demo Video
[Watch the demo here]https://www.loom.com/share/fc217ff8855445aeb1e1defa6098a7cb?sid=4ed5d592-fd7e-4ae0-81f2-eb69cf8937b3)

---

## Prerequisites
- Python 3.8+
- Google Generative AI API Key  (stored in a `config.ini` file)
- Required Python libraries (listed in `requirements.txt`)

---

## Setup and Execution Steps
### 1. Clone the Repository
```bash
$ git clone <repository-url>
$ cd <repository-folder>
```

### 2. Create and Activate a Virtual Environment
```bash
$ python -m venv env
$ source env/bin/activate    # On Windows: env\Scripts\activate
```

### 3. Install Dependencies
```bash
$ pip install -r requirements.txt
```

### 4. Configure API Keys
1. Create a `config.ini` file in the root folder.
2. Add your Google Gemini API key in the following format:
```ini
[google_api]
key = YOUR_API_KEY
```

### 5. Run the Flask Backend
```bash
$ python app.py
```


---

## Example Workflow
1. Upload one or more documents via the simple UI.
2. Enter a query in the UI 
3. The system retrieves the most relevant document chunks, uses Google Generative AI API to generates a response using RAG, and displays the answer.

---

## Future Enhancements
- Add support for multi-language queries.
- Introduce more advanced LLMs for improved accuracy and context handling.
- Build a more dynamic and interactive UI for enhanced user experience.

---

## Acknowledgements
Special thanks to the hackathon organizers and the department for providing us with this opportunity to innovate, collaborate, and learn. 


