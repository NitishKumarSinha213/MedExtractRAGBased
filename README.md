⚕️ MedExtract AI: Agentic RAG Extractor
MedExtract AI is a high-precision medical data extraction framework built with Java 21, Spring Boot, and LangChain4j. It utilizes a "Small-to-Big" (Parent Document) Retrieval-Augmented Generation (RAG) pattern to ensure medical headers (like Patient Names) are never lost during semantic search, even when using local LLMs.

Demo Video : https://drive.google.com/file/d/1lXYIEPJMYdFFkGdGa9h9DIo7ODWmR6n8/view?usp=sharing

🚀 Key Features
1. Parent Document Retrieval: Decouples search precision from LLM context. Uses 300-character "child" chunks for high-accuracy vector matching and 3000-character "parent" blocks for full clinical context.
2. Local-First Intelligence: Powered by Ollama (Qwen 2.5 7B) and BioBERT embeddings—ensuring patient data never leaves your local environment.
3. Ephemeral Vector Memory: Implements per-request InMemoryEmbeddingStore to prevent cross-contamination between different patient files.
4. Structured PDF Parsing: Uses PDFTextStripper with positional sorting to maintain the integrity of medical record headers and tables.
5. Modern Workspace UI: A three-pane React dashboard for simultaneous control, AI reasoning, and source verification.

🛠️ Technical Stack
Backend: Java 21, Spring Boot 3.x
AI Orchestration: LangChain4j
LLM: Qwen 2.5 (via Ollama)
Embeddings: AllMiniLmL6V2 / BioBERT
Frontend: React.js (Axios for API communication)
Parsing: Apache PDFBox

📂 Architecture
The system follows an Agentic Extraction Pipeline:

Ingestion: PDF is stripped with positional awareness and cleaned of whitespace noise.
Hierarchy Creation: The document is split into 3000-character parents. Each parent is subdivided into 300-character children.
Vectorization: Only child chunks are embedded into the ephemeral store, tagged with a parent_id.
Retrieval: The system searches for the top 3 children, then "hydrates" the prompt by fetching their full 3000-character parent segments.
Grounded Generation: The LLM is forced via @SystemMessage to use only the retrieved context, preventing medical hallucinations.
