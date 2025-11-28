# Java Restaurant Chatbot

**Project Contributors:**
 Selmah Tzindori- Data Collection Assist,Data Preparation and EDA,Evaluation of the Chatbot
 Saruni Mores- Data Collection Lead,Chatbot Developer & Workflow Engineer

---

## Project Overview
The **Java Restaurant Chatbot** is an AI-powered conversational agent designed to assist customers in interacting with the Java Restaurant. It handles queries such as:

- Menu items, prices, and specials  
- Reservation assistance  
- Opening hours and general restaurant information  
- Personalized customer service

The project combines a **knowledge base** approach with potential AI/NLP enhancements for improved user experience.

---

## Project Workflow

The project workflow is structured in the following stages:

### 1. Data Collection
- Manual collection of restaurant-related data (menu items, prices, reservation info, FAQs, etc.)  
- Sources include:
  - Restaurant internal records
  - Publicly available information
  - Sample conversation data for chatbot training

### 2. Data Storage
- Data is stored in structured formats (CSV, JSON) suitable for ingestion by the chatbot.  
- Knowledge base files are organized under the `data/` directory.

### 3. Data Preprocessing
- Clean and format data for consistency:
  - Handle missing values  
  - Standardize text formatting  
  - Remove duplicates and irrelevant entries

### 4. Exploratory Data Analysis (EDA)
EDA is performed to understand the datasets and prepare them for chatbot integration. Key steps include:

- **Descriptive statistics:** Understand distributions of menu items, prices, and reservation patterns.  
- **Data visualization:** Charts and plots for frequent queries, popular menu items, and peak hours.  
- **Data validation:** Ensure accuracy and completeness of knowledge base entries.  

Tools used for EDA may include Python libraries like `pandas`, `matplotlib`, or `seaborn`.

### 5. Knowledge Base Construction
- A structured knowledge base is built using cleaned data.  
- Key components:
  - FAQs  
  - Menu catalog  
  - Reservation rules  
- This forms the core reference for the chatbot’s responses.

### 6. Chatbot Development
- Integration of the knowledge base into the chatbot workflow (n8n or other AI frameworks).  
- Chatbot can be accessed through different channels (web interface, Telegram, WhatsApp).  
- Initial training focuses on:
  - Matching user queries to KB entries  
  - Handling common conversation intents  
  - Providing accurate responses

### 7. Model Training & Evaluation (Optional / Future)
- If implementing NLP models for improved conversation:
  - Train models on historical query-response data  
  - Evaluate using metrics such as accuracy, F1-score, or conversational success rate  
  - Fine-tune the knowledge base and response patterns based on evaluation

### 8. Deployment
- Currently, the chatbot is run locally for testing and debugging.  
- Future deployment plans include:
  - Hosting on a cloud server  
  - Connecting to messaging platforms via API  
  - Ensuring secure access and data privacy

---



