# tds-project1
A virtual TA that leverages hybrid RAG to answer student questions. It uses discourse posts and webpage contents as sources of information and LLM to generate answers along with cited sources.

## Data Preprocessing
IITM TDS discourse posts from 1st January, 2025 to 15th April, 2025 were scraped using the discourse API. Course content on the TDS course page for the January 2025 term were retreived using the public github repository. 
- fetch-discourse.py retrieves the discourse posts as mentioned.
- clean-discourse.py removes PII, extracts images to encode them as base64 urls and uses BeautifulSoup to remove HTML tags.
- fetch-course-content.py retrieves markdown files from the locally cloned public TDS repository. 
- clean-course-content.py removes comment tags, regularizes whitespaces and extracts youtube links from the text. 

The resulting cleaned data is stored in cleaned-discourse.json and cleaned-content.json files. 

## Embedding
Nomic Atlas API is used to embed text and images. This is stored in an SQLite Database. 