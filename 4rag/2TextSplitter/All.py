"""
🧠 LangChain TextSplitter — Document Chunking for LLM Workflows

TextSplitter in LangChain is a core utility that splits large documents into smaller, manageable chunks that are compatible with token limits and ideal for downstream LLM tasks like RAG, summarization, and embedding generation.

===============================================================================
🔹 WHY Split Text?
===============================================================================
✅ Language models have token limits — exceeding them causes failures or truncation  
✅ Proper chunking improves semantic search, reduces hallucination, and boosts accuracy  
✅ Helps with streaming, batch processing, and parallel inference  

===============================================================================
🔹 📦 Available TextSplitters in LangChain
===============================================================================

1. ✅ **RecursiveCharacterTextSplitter**  
   - Smartest and most commonly used splitter  
   - Breaks based on logical structure: `["\n\n", "\n", ".", " "]`  
   - Preserves context while chunking  
   - Ideal for plain text, PDFs, articles, etc.

2. ✂️ **CharacterTextSplitter**  
   - Basic splitter using a single separator like `\n` or space  
   - Not structure-aware; splits blindly  
   - Useful for custom low-level splitting logic  

3. 🔢 **TokenTextSplitter**  
   - Splits text based on token count (not character count)  
   - Depends on tokenizer (e.g., tiktoken or HuggingFace)  
   - Best when working with token-sensitive LLMs like GPT  

4. 🧱 **MarkdownHeaderTextSplitter**  
   - Specialized for markdown files  
   - Retains hierarchy of headers (`#`, `##`, etc.)  
   - Great for structured documents like technical wikis or README.md files

5. 📄 **DocumentTextSplitter**  
   - Works on multiple documents (not just raw text)  
   - Can group small docs into one chunk or split big docs into many  
   - Useful for chunking in multi-file processing pipelines  

6. 🧠 **SemanticChunker** (LangChain experimental)  
   - Uses embeddings to split text where **semantic shift** occurs  
   - Provides the **most contextually accurate** chunks  
   - Requires an embedding model and a vector distance metric  
   - Ideal for tasks like semantic search or long-document summarization  

7. 📏 **LengthBasedSplitter**  
   - Simply chunks based on fixed lengths (e.g., every 1000 characters)  
   - Fast but no understanding of semantics or sentence boundaries  
   - Good for logs or fixed-format data  

===============================================================================
🔹 ⚙️ Common Parameters
===============================================================================
• `chunk_size`: Max length of each chunk (chars or tokens)  
• `chunk_overlap`: Amount of overlap between chunks  
• `separators`: Custom list of split points for Recursive/Character  
• `length_function`: Function to calculate chunk length (token or char-based)

===============================================================================
🔹 ✅ Example: RecursiveCharacterTextSplitter
===============================================================================
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)

chunks = text_splitter.split_documents(documents)

===============================================================================
🔹 ✅ Example: SemanticChunker (experimental)
===============================================================================
from langchain.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
semantic_splitter = SemanticChunker(embeddings=embedding_model, breakpoint_threshold_type="percentile")

chunks = semantic_splitter.split_documents(documents)

===============================================================================
🔹 ✅ Example: TokenTextSplitter
===============================================================================
from langchain.text_splitter import TokenTextSplitter

token_splitter = TokenTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

chunks = token_splitter.split_text(text)

===============================================================================
✅ RECOMMENDATION:
- Use **RecursiveCharacterTextSplitter** by default  
- Use **TokenTextSplitter** for token-constrained models like OpenAI  
- Use **SemanticChunker** if semantic accuracy is critical  
- Avoid splitting arbitrarily; use overlapping chunks to preserve context  

===============================================================================
🎯 Summary:
LangChain's TextSplitter module is essential for efficient LLM input preparation — it ensures your documents are split in a context-aware, size-limited manner to maximize model effectiveness in production-grade pipelines.

"""
