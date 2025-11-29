# Java Restaurant Chatbot

**Project Contributors:**
1) Selmah Tzindori- Data Collection Assist,Data Preparation and EDA, Chatbot Assist & Evaluation of the Chatbot
 
2) Saruni Mores- Data Collection Lead,Chatbot Developer & Workflow Engineer

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
- Knowledge base files are organized under the restaurant_kb files(.json &csv)

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

### 7. Chatbot Evaluation

To ensure the Java Restaurant Chatbot performs effectively, we evaluate it across the following metrics:

1. Accuracy
- Test queries against expected answers from the knowledge base.
- Score: Correct = 1, Partial = 0.5, Incorrect = 0.
- Accuracy (%) = (Correct Responses / Total Test Questions) × 100

2. Coverage
- Measure the percentage of knowledge base entries correctly triggered by test queries.
- Coverage (%) = (KB entries correctly triggered / Total KB entries) × 100

3. Usability
- Assess clarity, tone, and response handling by testers.
- Feedback scored on a scale of 1–5.

4. Robustness
- Test multiple rapid queries and edge cases to ensure stability.

5. Future NLP Metrics
- If NLP models are added in the future: evaluate Precision, Recall, F1-score, and response appropriateness.

---

## Future Recommendations and Improvements

- **NLP Integration:** Implement natural language understanding to handle more diverse user queries and improve conversational flow.  
- **Multi-Platform Deployment:** Deploy the chatbot on cloud servers and integrate with messaging platforms like WhatsApp, Telegram, and web interfaces.  
- **Dynamic Knowledge Base:** Enable real-time updates to menu items, specials, and reservations without redeploying the bot.  
- **Analytics Dashboard:** Track user interactions, frequently asked questions, and chatbot performance for continuous improvement.  
- **Personalization:** Add user profiles to provide tailored recommendations and offers.  

## Challenges Encountered

- **Data Collection:** Manual collection of restaurant data and FAQs was time-consuming.  
- **Knowledge Base Maintenance:** Ensuring consistency and coverage of all possible queries was challenging.  
- **Local Deployment Limitations:** Current chatbot runs locally, limiting accessibility and testing across multiple devices.  
- **Handling Edge Cases:** Uncommon or poorly formatted queries sometimes resulted in incorrect or empty responses.  
- **Scalability:** Preparing the system for higher traffic and future integration with AI/NLP models requires additional resources.

---

### Chatbot URL
```
https://huggingface.co/spaces/SelmahT/java_restaurant
```

---


