## HCL_TECH_ENDGAME

# Legal Document Analyser using RAG 

A “Legal Document Analyser” is a tool (or a structured method) that helps you understand any legal document quickly and clearly.


Tech Stacks using - LLM models (free model via hugging face mostly or gemini models using api )
if our system not work well - we do quantization using Lora/Qlora finetuning llm

## Methodology

-load constitution data (books , pdf , reserach paper etc )
-then we do document chuncking (documentloder for documentload)
-we cteate as function for case breif - genrate case breif 
-text chunking - we chunk larger data into smaller chunks 

-we do store  embedding using vector database(**A futuristic digital illustration of a legal AI assistant. A glowing digital version of a 
Constitution book is open in the center, with streams of binary code and glowing nodes connecting specific legal articles to a chatbot 
interface on a glass tablet. The background is a clean, professional dark blue with cyber-security aesthetics. High tech, detailed, isometric view.**)

-then load embedding model 
-then we create a function to genrete embedding - then we crete a function to identify the most relevent docment to resopnse 

we use Rag to retreive the most relevent chunk 

#workflow ->
#upload pdf -> processing ->embedding and processing ->strore in vector db ->query handling using rag -> using similarity serch to get most relevent response->deploy (web ui , Cli)

#workflow
[workflow](![WhatsApp Image 2025-12-04 at 4 37 46 PM](https://github.com/user-attachments/assets/749810e3-5bf1-4acc-988d-068fe81119ab)





