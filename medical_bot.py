import customtkinter as ctk
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import os
import pygame
import threading
import time
import random

# Initialize Pygame Mixer for smooth audio
pygame.mixer.init()

# ==================================================================================
#                       BACKEND: THE AI BRAIN (Machine Learning)
# ==================================================================================

class DoctorBrain:
    def __init__(self):
        # The Knowledge Base
        # We map "Patterns" (User inputs) to "Responses" (Doctor answers)
        self.medical_data = {
            "greeting": {
                "patterns_en": "hello hi hey doctor good morning good evening help me wake up",
                "patterns_hi": "नमस्ते हेलो डॉक्टर मदद करो कैसे हो सुप्रभात",
                "response_en": "Hello! I am Dr. AI. I am ready to help. Please describe your symptoms.",
                "response_hi": "नमस्ते! मैं डॉ. एआई हूँ। बताईये आपको क्या समस्या हो रही है?"
            },
            "fever": {
                "patterns_en": "i have fever high temperature body is burning shivering feeling hot cold chills thermometer reads high",
                "patterns_hi": "मुझे बुखार है मेरा शरीर गरम है कंपकंपी हो रही है तापमान ज्यादा है ठंड लग रही है",
                "response_en": "Diagnosis: Viral Fever.\nRx: Take Paracetamol 650mg, use a cold sponge on your forehead, and drink plenty of fluids.",
                "response_hi": "निदान: वायरल बुखार।\nइलाज: पैरासिटामोल 650mg लें, माथे पर गीली पट्टी रखें और खूब पानी पिएं।"
            },
            "cold": {
                "patterns_en": "running nose sneezing cough sore throat blocked nose mucus phlegm congestion flu",
                "patterns_hi": "नाक बह रही है छींक आ रही है खांसी है गला खराब है जुकाम है बलगम",
                "response_en": "Diagnosis: Common Cold.\nRx: Steam inhalation 3 times a day. Gargle with salt water. Drink warm turmeric milk.",
                "response_hi": "निदान: सामान्य सर्दी।\nइलाज: दिन में 3 बार भाप लें। नमक के पानी से गरारे करें। हल्दी वाला दूध पिएं।"
            },
            "stomach": {
                "patterns_en": "stomach pain belly ache loose motions diarrhea vomiting acidity gas burning sensation digestion problem food poisoning",
                "patterns_hi": "पेट में दर्द है दस्त लगे हैं उल्टी हो रही है गैस बन रही है पेट खराब है एसिडिटी",
                "response_en": "Diagnosis: Gastric Issue.\nRx: Eat light food (Curd-Rice/Toast). Drink ORS solution. Avoid spicy and oily food.",
                "response_hi": "निदान: पेट की समस्या।\nइलाज: हल्का खाना (दही-चावल) खाएं। ORS का घोल पिएं। मसालेदार खाने से बचें।"
            },
            "headache": {
                "patterns_en": "headache splitting head pain migraine dizzy heavy head stress tension",
                "patterns_hi": "सिर दर्द कर रहा है सिर भारी है माइग्रेन चक्कर आ रहे हैं तनाव",
                "response_en": "Diagnosis: Tension Headache.\nRx: Drink a large glass of water immediately. Rest in a dark, quiet room. Stay off screens.",
                "response_hi": "निदान: सिरदर्द।\nइलाज: तुरंत एक गिलास पानी पिएं। अंधेरे और शांत कमरे में आराम करें। फोन का इस्तेमाल न करें।"
            },
            "thanks": {
                "patterns_en": "thank you thanks bye goodbye see you exit quit",
                "patterns_hi": "धन्यवाद शुक्रिया बाय अलविदा टाटा",
                "response_en": "You're welcome! Take care of your health. Goodbye!",
                "response_hi": "स्वागत है! अपनी सेहत का ख्याल रखें। अलविदा!"
            }
        }

        # --- Train the AI (Vectorization) ---
        self.labels = list(self.medical_data.keys())
        
        # 1. English Training
        self.corpus_en = [self.medical_data[k]["patterns_en"] for k in self.labels]
        self.vectorizer_en = TfidfVectorizer()
        self.tfidf_matrix_en = self.vectorizer_en.fit_transform(self.corpus_en)

        # 2. Hindi Training
        self.corpus_hi = [self.medical_data[k]["patterns_hi"] for k in self.labels]
        self.vectorizer_hi = TfidfVectorizer()
        self.tfidf_matrix_hi = self.vectorizer_hi.fit_transform(self.corpus_hi)

    def get_prediction(self, user_text, lang='en'):
        """Computes mathematical similarity between user input and database"""
        
        # Select Language Model
        if lang == 'en':
            vectorizer = self.vectorizer_en
            matrix = self.tfidf_matrix_en
        else:
            vectorizer = self.vectorizer_hi
            matrix = self.tfidf_matrix_hi

        # Convert user text to numbers
        user_tfidf = vectorizer.transform([user_text])
        
        # Calculate Cosine Similarity (0 to 1)
        similarities = cosine_similarity(user_tfidf, matrix)
        
        # Find best match
        best_match_idx = similarities.argmax()
        score = similarities[0][best_match_idx]

        # AI Confidence Threshold (Must be > 20% sure)
        if score < 0.2:
            return None
        
        condition = self.labels[best_match_idx]
        
        # Return response text
        if lang == 'en':
            return self.medical_data[condition]["response_en"]
        else:
            return self.medical_data[condition]["response_hi"]


# ==================================================================================
#                       FRONTEND: THE MODERN GUI (CustomTkinter)
# ==================================================================================

class SmartDoctorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Configuration ---
        self.title("Smart Doctor AI 🩺")
        self.geometry("800x650")
        
        # Theme Settings
        ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
        ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

        # --- Initialize Backend ---
        self.brain = DoctorBrain()
        self.lang = "en"  # Default Language
        self.is_listening = False

        # --- Layout Grid Config ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # 1. Header Frame
        self.header_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="#2B2B2B")
        self.header_frame.grid(row=0, column=0, sticky="ew")
        
        self.title_label = ctk.CTkLabel(
            self.header_frame, 
            text="🩺 VIRTUAL HEALTH ASSISTANT", 
            font=("Segoe UI", 24, "bold"),
            text_color="#4CC9F0"
        )
        self.title_label.pack(pady=15)

        # 2. Chat History Area
        self.chat_display = ctk.CTkTextbox(
            self, 
            font=("Roboto Medium", 14), 
            text_color="white",
            fg_color="#1E1E1E",
            wrap="word"
        )
        self.chat_display.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="nsew")
        self.chat_display.insert("end", "🩺 Doctor: Hello! How can I help you today?\n\n")
        self.chat_display.configure(state="disabled") # Read only

        # 3. Controls Area (Bottom)
        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=2, column=0, padx=20, pady=20, sticky="ew")
        self.controls_frame.grid_columnconfigure(0, weight=1)

        # Input Box
        self.entry_box = ctk.CTkEntry(
            self.controls_frame, 
            placeholder_text="Type your symptoms here...", 
            height=50, 
            font=("Arial", 14),
            corner_radius=25,
            border_color="#4CC9F0"
        )
        self.entry_box.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.entry_box.bind("<Return>", self.on_enter_pressed)

        # Voice Button
        self.voice_btn = ctk.CTkButton(
            self.controls_frame, 
            text="🎤 Speak", 
            width=100, 
            height=50, 
            corner_radius=25,
            fg_color="#E63946", 
            hover_color="#D62828",
            font=("Arial", 14, "bold"),
            command=self.start_listening_thread
        )
        self.voice_btn.grid(row=0, column=1, padx=(0, 10))

        # Send Button
        self.send_btn = ctk.CTkButton(
            self.controls_frame, 
            text="➤ Send", 
            width=100, 
            height=50, 
            corner_radius=25,
            fg_color="#4361EE", 
            hover_color="#3A0CA3",
            font=("Arial", 14, "bold"),
            command=self.handle_message
        )
        self.send_btn.grid(row=0, column=2, padx=(0, 10))

        # Language Toggle
        self.lang_btn = ctk.CTkButton(
            self.controls_frame,
            text="A/क",
            width=50,
            height=50,
            corner_radius=25,
            fg_color="#F72585",
            command=self.toggle_language
        )
        self.lang_btn.grid(row=0, column=3)


    # --- GUI INTERACTIONS ---

    def toggle_language(self):
        if self.lang == "en":
            self.lang = "hi"
            self.add_message("System", "Language switched to Hindi (हिंदी)", "sys")
            self.speak_response("भाषा हिंदी में बदल दी गई है")
        else:
            self.lang = "en"
            self.add_message("System", "Language switched to English", "sys")
            self.speak_response("Language switched to English")

    def add_message(self, sender, message, tag):
        """Updates the Chat Window safely"""
        self.chat_display.configure(state="normal")
        
        if tag == "user":
            formatted_msg = f"\n👤 You: {message}\n"
        elif tag == "bot":
            formatted_msg = f"\n🩺 Doctor: {message}\n"
        else:
            formatted_msg = f"\n[SYSTEM]: {message}\n"
            
        self.chat_display.insert("end", formatted_msg)
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end") # Auto scroll down

    def on_enter_pressed(self, event):
        self.handle_message()

    def handle_message(self):
        user_text = self.entry_box.get()
        if not user_text.strip():
            return
            
        # 1. Clear input
        self.entry_box.delete(0, "end")
        
        # 2. Show user message
        self.add_message("You", user_text, "user")
        
        # 3. Process in Thread (Prevent Freeze)
        threading.Thread(target=self.process_ai, args=(user_text,)).start()

    # --- AI & AUDIO PROCESSING ---

    def process_ai(self, text):
        # Think...
        response = self.brain.get_prediction(text.lower(), self.lang)
        
        if not response:
            if self.lang == 'hi':
                response = "क्षमा करें, मुझे समझ नहीं आया। कृपया स्पष्ट रूप से बताएं (जैसे: 'मुझे बुखार है')."
            else:
                response = "I'm not sure I understand. Please describe symptoms like 'Fever', 'Headache', etc."
        
        # Display response
        self.add_message("Doctor", response, "bot")
        
        # Speak response
        self.speak_response(response)

    def speak_response(self, text):
        """Generates and plays audio without freezing the app"""
        try:
            lang_code = 'hi' if self.lang == 'hi' else 'en'
            tld_code = 'co.in' # Indian accent
            
            # Use random filename to avoid permission locks
            filename = f"voice_{random.randint(1000, 99999)}.mp3"
            
            # Generate
            tts = gTTS(text=text, lang=lang_code, tld=tld_code)
            tts.save(filename)
            
            # Play via Pygame Mixer
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            
            # Wait loop (Background thread)
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            # Cleanup
            pygame.mixer.music.unload()
            try:
                os.remove(filename)
            except:
                pass
                
        except Exception as e:
            print(f"Audio Error: {e}")

    # --- VOICE RECOGNITION ---

    def start_listening_thread(self):
        self.voice_btn.configure(fg_color="#00FF00", text="Listening...")
        threading.Thread(target=self.listen_voice).start()

    def listen_voice(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            try:
                r.adjust_for_ambient_noise(source, duration=0.5)
                # Listen
                audio = r.listen(source, timeout=5, phrase_time_limit=10)
                
                # Transcribe
                lang_code = "hi-IN" if self.lang == 'hi' else "en-IN"
                text = r.recognize_google(audio, language=lang_code)
                
                # Send to processing
                self.add_message("You", text, "user")
                threading.Thread(target=self.process_ai, args=(text,)).start()
                
            except sr.WaitTimeoutError:
                self.add_message("System", "No speech detected.", "sys")
            except sr.UnknownValueError:
                self.add_message("System", "Could not understand audio.", "sys")
            except Exception as e:
                print(e)
            
            # Reset button
            self.voice_btn.configure(fg_color="#E63946", text="🎤 Speak")


# ==================================================================================
#                                 MAIN EXECUTION
# ==================================================================================

if __name__ == "__main__":
    app = SmartDoctorApp()
    app.mainloop()
