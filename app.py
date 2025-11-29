# app.py
import json
import csv
from pathlib import Path
import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
import threading
import pandas as pd

# -------- CONFIG ----------
KB_FILE_JSON = "restaurant_kb.json"
KB_FILE_CSV = "restaurant_kb.csv"
EMBED_CACHE = "kb_embeddings.npz"
CONFIDENCE_THRESHOLD = 0.40
LOG_FILE = "logs.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_SUGGESTIONS = 3  # Maximum similar suggestions for KB fallback
AUTOCOMPLETE_LIMIT = 5  # Suggestions while typing
# --------------------------

# Load model
model = SentenceTransformer(MODEL_NAME)

# -------- LOAD KB + FAQ --------
def load_kb(file_json=KB_FILE_JSON, file_csv=KB_FILE_CSV):
    kb = []
    if Path(file_json).exists():
        with open(file_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
            for item in raw:
                kb.append({
                    "question": item.get("Name / Question") or item.get("Name") or "",
                    "answer": item.get("Details / Answer") or item.get("Details") or "",
                    "price": item.get("Price_KES") or "",
                    "prep_time": item.get("Estimated_Prep_Time_mins") or "",
                    "diet": item.get("Tags_Dietary_Info") or "",
                    "age_tag": item.get("Tags_Age") or ""
                })
    elif Path(file_csv).exists():
        with open(file_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                kb.append({
                    "question": row.get("Name / Question") or row.get("Name") or "",
                    "answer": row.get("Details / Answer") or row.get("Details") or "",
                    "price": row.get("Price_KES") or "",
                    "prep_time": row.get("Estimated_Prep_Time_mins") or "",
                    "diet": row.get("Tags_Dietary_Info") or "",
                    "age_tag": row.get("Tags_Age") or ""
                })
    else:
        raise FileNotFoundError(f"No knowledge base file found at {file_json} or {file_csv}.")

    # Add FAQ entries
    faq_entries = [
        {"question": "What are your opening hours?", "answer": "⏰ We are open daily from 9 AM to 11 PM.", "price":"", "prep_time":"", "diet":"", "age_tag":""},
        {"question": "Do you offer delivery?", "answer": "🚚 We deliver within Nairobi. Delivery charges may apply.", "price":"", "prep_time":"", "diet":"", "age_tag":""},
        {"question": "Can I reserve a table?", "answer": "📞 You can reserve a table by calling us or asking here.", "price":"", "prep_time":"", "diet":"", "age_tag":""},
        {"question": "Do you have vegetarian options?", "answer": "Yes! We have several vegetarian dishes like Veggie Burger, Potato Wedges, and Mushroom Chips.", "price":"", "prep_time":"", "diet":"", "age_tag":""},
        {"question": "What is your contact number?", "answer": "📧 You can reach us at java@gmail.com or call +254 722 123 456.", "price":"", "prep_time":"", "diet":"", "age_tag":""},
        {"question": "Do you have Wi-Fi?", "answer": "Yes, our restaurant has free Wi-Fi for all customers.", "price":"", "prep_time":"", "diet":"", "age_tag":""},
        {"question": "Are pets allowed?", "answer": "Service animals only. No other pets are allowed inside.", "price":"", "prep_time":"", "diet":"", "age_tag":""},
        {"question": "Payment methods?", "answer": "We accept cash, cards, and M-Pesa.", "price":"", "prep_time":"", "diet":"", "age_tag":""},
        {"question": "Do you have outdoor seating?", "answer": "Yes, we have a comfortable outdoor seating area.", "price":"", "prep_time":"", "diet":"", "age_tag":""},
        {"question": "Can I order for pickup?", "answer": "Yes, you can order ahead and pick up your order at the restaurant.", "price":"", "prep_time":"", "diet":"", "age_tag":""}
    ]
    kb.extend(faq_entries)
    return kb

# -------- EMBEDDINGS --------
def prepare_embeddings(kb, cache_path=EMBED_CACHE):
    questions = [item["question"] for item in kb]
    if Path(cache_path).exists():
        try:
            data = np.load(cache_path, allow_pickle=True)
            cached_questions = data["questions"].tolist()
            if cached_questions == questions:
                embeddings = data["embeddings"]
                return questions, embeddings
        except Exception:
            pass
    embeddings = model.encode(questions, show_progress_bar=True)
    np.savez_compressed(cache_path, questions=questions, embeddings=embeddings)
    return questions, embeddings

kb_data = load_kb()
kb_questions, kb_embeddings = prepare_embeddings(kb_data)

# -------- LOGGING --------
log_lock = threading.Lock()
def log_interaction(user, response, confidence):
    exists = Path(LOG_FILE).exists()
    with log_lock:
        df = pd.DataFrame([{
            "timestamp": datetime.utcnow().isoformat(),
            "user_message": user,
            "bot_response": response,
            "confidence": float(confidence)
        }])
        if exists:
            df.to_csv(LOG_FILE, mode="a", header=False, index=False)
        else:
            df.to_csv(LOG_FILE, mode="w", header=True, index=False)

# -------- AUTOCOMPLETE FUNCTION --------
def get_autocomplete_suggestions(user_input):
    user_lower = user_input.lower()
    matches = [q for q in kb_questions if user_lower in q.lower()]
    return matches[:AUTOCOMPLETE_LIMIT]

# -------- RESPONSE FUNCTION --------
def get_response(user_input):
    user_input_lower = user_input.lower().strip()

    # Greetings / farewells / thanks
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon"]
    farewells = ["bye", "goodbye", "see you"]
    thanks = ["thank you", "thanks", "thx"]

    if any(word in user_input_lower for word in greetings):
        return "👋 Hello! Welcome to **Java Restaurant**. How can I help you today?", 1.0
    if any(word in user_input_lower for word in farewells):
        return "👋 Goodbye! Hope to see you soon at Java Restaurant.", 1.0
    if any(word in user_input_lower for word in thanks):
        return "😊 You’re welcome! Anything else I can help with?", 1.0

    # Trigger keywords (menu, kids menu, delivery, contact, etc.)
    menu_keywords = ["menu", "our menu", "what do you serve", "food options"]
    kids_menu_keywords = ["kids menu", "menu for kids", "children menu"]
    opening_hours_keywords = ["opening hours", "hours", "open", "close", "time"]
    reservations_keywords = ["reserve", "reservation", "book a table"]
    delivery_keywords = ["delivery", "deliver", "home delivery"]
    contact_keywords = ["contact", "phone", "number", "email"]

    if any(kw in user_input_lower for kw in menu_keywords):
        menu_items = [item for item in kb_data if item['price'] or item['prep_time']]
        response_lines = []
        for item in menu_items:
            line = f"🍽 {item['question']}: {item['answer']}"
            flair_parts = []
            if item['price']: flair_parts.append(f"💰 {item['price']}")
            if item['prep_time']: flair_parts.append(f"⏱ {item['prep_time']} mins")
            if item['diet']: flair_parts.append(f"🥗 {item['diet']}")
            if flair_parts:
                line += " | " + " | ".join(flair_parts)
            response_lines.append(line)
        menu_response = "<br>".join(response_lines)
        log_interaction(user_input, menu_response, 1.0)
        return menu_response, 1.0

    if any(kw in user_input_lower for kw in kids_menu_keywords):
        kids_items = [item for item in kb_data if item.get("age_tag") == "Kids"]
        if not kids_items:
            kids_items = [item for item in kb_data if "Mini" in item['question'] or "Kids" in item['question']]
        response_lines = []
        for item in kids_items:
            line = f"🍭 {item['question']}: {item['answer']}"
            flair_parts = []
            if item['price']: flair_parts.append(f"💰 {item['price']}")
            if item['prep_time']: flair_parts.append(f"⏱ {item['prep_time']} mins")
            if item['diet']: flair_parts.append(f"🥗 {item['diet']}")
            if flair_parts:
                line += " | " + " | ".join(flair_parts)
            response_lines.append(line)
        kids_menu_response = "<br>".join(response_lines)
        log_interaction(user_input, kids_menu_response, 1.0)
        return kids_menu_response, 1.0

    if any(kw in user_input_lower for kw in opening_hours_keywords):
        response = "⏰ We are open daily from 9 AM to 11 PM."
        log_interaction(user_input, response, 1.0)
        return response, 1.0
    if any(kw in user_input_lower for kw in reservations_keywords):
        response = "📞 You can reserve a table by calling us at +254 700 000 000 or asking here."
        log_interaction(user_input, response, 1.0)
        return response, 1.0
    if any(kw in user_input_lower for kw in delivery_keywords):
        response = "🚚 We offer delivery within Nairobi. Delivery charges may apply."
        log_interaction(user_input, response, 1.0)
        return response, 1.0
    if any(kw in user_input_lower for kw in contact_keywords):
        response = "📧 You can reach us at java@example.com or call +254 700 000 000."
        log_interaction(user_input, response, 1.0)
        return response, 1.0

    # -------- KB EMBEDDING FALLBACK WITH SUGGESTIONS --------
    user_emb = model.encode([user_input])
    sims = cosine_similarity(user_emb, kb_embeddings)[0]
    best_indices = sims.argsort()[::-1][:MAX_SUGGESTIONS]
    best_score = float(sims[best_indices[0]])
    best_item = kb_data[best_indices[0]]

    if best_score < CONFIDENCE_THRESHOLD:
        # Suggest similar items
        suggestions = [kb_data[idx]['question'] for idx in best_indices if sims[idx] > 0.2]
        if suggestions:
            fallback = "🤔 I'm not sure about that. Did you mean: " + ", ".join(suggestions) + "?"
        else:
            fallback = "🤔 I'm not sure about that. Try asking about menu items, prices, prep time, or dietary info! 🍟"
        log_interaction(user_input, fallback, best_score)
        return fallback, best_score

    # Construct answer with flair
    flair_parts = []
    if best_item['price']: flair_parts.append(f"💰 {best_item['price']}")
    if best_item['prep_time']: flair_parts.append(f"⏱ {best_item['prep_time']} mins")
    if best_item['diet']: flair_parts.append(f"🥗 {best_item['diet']}")
    flair_text = " | ".join(flair_parts)

    if best_item['price'] or best_item['prep_time'] or best_item['diet']:
        answer = f"<div style='border:2px solid #FF5733; padding:10px; border-radius:10px; background-color:#FFF5F0'>"
        answer += f"<b>{best_item['question']}</b><br>{best_item['answer']}"
        if flair_text:
            answer += f"<br><i>{flair_text}</i>"
        answer += "</div>"
    else:
        answer = best_item['answer']

    log_interaction(user_input, answer, best_score)
    return answer, best_score

# -------- GRADIO UI WITH AUTOCOMPLETE --------
with gr.Blocks(title="Java Restaurant Chatbot") as demo:
    gr.Markdown("<h1 style='color:#FF5733'>🍽 Java Restaurant Chatbot 🍽</h1><hr>")
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(elem_id="chatbot", label="Chat")
            msg = gr.Textbox(placeholder="Ask about menu, reservations, hours...", show_label=False)
            submit = gr.Button("Send")
            clear_btn = gr.Button("Clear Chat")
            autocomplete_box = gr.Dropdown(label="Suggestions", interactive=True, choices=[], multiselect=False)

    # Update autocomplete suggestions dynamically
    def update_autocomplete(user_text):
        suggestions = get_autocomplete_suggestions(user_text)
        return gr.Dropdown.update(choices=suggestions)

    msg.change(fn=update_autocomplete, inputs=[msg], outputs=[autocomplete_box])

    # When user clicks suggestion, fill textbox
    def fill_from_suggestion(choice):
        return choice

    autocomplete_box.change(fn=fill_from_suggestion, inputs=[autocomplete_box], outputs=[msg])

    def user_submit(user_text, history):
        answer, conf = get_response(user_text)
        history = history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer}
        ]
        return history, ""  # clear input

    submit.click(fn=user_submit, inputs=[msg, chatbot], outputs=[chatbot, msg])
    msg.submit(fn=user_submit, inputs=[msg, chatbot], outputs=[chatbot, msg])
    clear_btn.click(lambda: [], None, chatbot)

    gr.Markdown("**Deployment**: After local testing, push to GitHub or create a Hugging Face Space.")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7864, share=True)
