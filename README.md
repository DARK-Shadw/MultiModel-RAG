# PDFLoader

A simple Python class to load and process PDF documents, extracting text, tables, and images, and creating summaries for retrieval-augmented generation (RAG) workflows.

---

## Purpose

`PDFLoader` processes PDF files by:
- Extracting text, tables, and images.
- Generating summaries for each element using a large language model (LLM).
- Storing the data in a vector store for retrieval using LangChain's `MultiVectorRetriever`.

It leverages Together AI for text and vision LLMs, Google Generative AI for embeddings, and Chroma for vector storage.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/pdf-to-podcast.git
   cd pdf-to-podcast
   ```

2. **Set Up Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install langchain-chroma langchain-community langchain-google-genai unstructured langchain-together python-dotenv
   ```
   **Note**: The `unstructured` library requires additional system dependencies (e.g., `poppler`, `tesseract`). Follow the [Unstructured Installation Guide](https://unstructured-io.github.io/unstructured/installing.html) for setup.

3. **Set Up Environment Variables**:
   Create a `.env` file:
   ```env
   TOGETHER_API_KEY=your_together_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```

---

## Usage

1. **Initialize the PDFLoader**:
   ```python
   from pdf_loader import PDFLoader

   # Path to your PDF file
   pdf_path = "PDFs/your_document.pdf"

   # Create an instance of PDFLoader
   loader = PDFLoader(pdf_path)
   ```

2. **Access Processed Data**:
   - Get text chunks:
     ```python
     texts = loader.gettexts()
     print(texts)
     ```
   - Get all chunks (text, tables, images):
     ```python
     chunks = loader.getchunks()
     print(chunks)
     ```

3. **Use the Retriever**:
   The `retriever` attribute (`loader.retriever`) can be used to search for relevant content:
   ```python
   query = "What is the main topic of the document?"
   results = loader.retriever.invoke(query)
   print(results)
   ```

---

## Project Structure

```
pdf-to-podcast/
├── PDFs/                 # Directory for input PDF files
├── pdf_loader.py         # PDFLoader class implementation
├── util.py               # Utility functions (e.g., check_exist)
├── requirements.txt      # Project dependencies
├── .env                  # Environment variables (not tracked)
├── PDFLoader_README.md   # This file
└── .gitignore            # Git ignore file
```

---

## Dependencies

- `langchain-chroma`
- `langchain-community`
- `langchain-google-genai`
- `unstructured`
- `langchain-together`
- `python-dotenv`

Install them using:
```bash
pip install -r requirements.txt
```

---

## Notes

- Ensure your PDF file is in the `PDFs/` directory or update the path.
- The script uses rate-limiting to avoid API limits; adjust `time.sleep` if needed.
- `util.py` should include required utility functions (e.g., `check_exist`).

---

## License

MIT License. See [LICENSE](LICENSE) for details.
