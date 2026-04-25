# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import glob
import face_recognition
import speech_recognition as sr
import time
import uuid
from gtts import gTTS
import pygame
from fer import FER
from ultralytics import YOLO
from playsound import playsound
import threading
import sys
import io
import datetime
import pywhatkit
import wikipediaapi
from googletrans import Translator
import tempfile
import parselmouth  # استيراد مكتبة parselmouth لاستخراج النغمة (pitch)

# تعيين ترميز الإخراج إلى UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# إعداد ويكيبيديا باللغة العربية مع User-Agent
wiki_wiki = wikipediaapi.Wikipedia(
    language='ar',
    user_agent='MyNourApp/1.0 (https://myapp.com; myemail@example.com)'
)

# تهيئة مترجم النص
translator = Translator()

def search_wikipedia(query):
    """البحث في ويكيبيديا عن السؤال المطلوب وإرجاع ملخص مختصر."""
    page = wiki_wiki.page(query)
    if page.exists():
        return page.summary[:500]  # إرجاع أول 500 حرف من الملخص
    else:
        return None

def speak_text(text, lang='ar'):
    """
    دالة موحدة لتحويل النص إلى كلام باستخدام gTTS والتشغيل عبر pygame.
    تم تحسينها لتقليل زمن الانتظار باستخدام فترات نوم قصيرة.
    """
    filename = f"temp_{uuid.uuid4()}.mp3"
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    # انتظار قصير لتقليل الـ busy-wait
    while pygame.mixer.music.get_busy():
        time.sleep(0.05)
    pygame.mixer.music.stop()
    time.sleep(0.05)
    try:
        os.remove(filename)
    except PermissionError:
        print(f"تحذير: لم أتمكن من حذف الملف {filename} لأنه قيد الاستخدام.")

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25
        pygame.mixer.init()

    def load_encoding_images(self, images_path):
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print(f"{len(images_path)} صورة تم العثور عليها.")
        for img_path in images_path:
            img = cv2.imread(img_path)
            if img is None:
                print(f"تحذير: لم أتمكن من قراءة الصورة {img_path}.")
                continue
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            encodings = face_recognition.face_encodings(rgb_img)
            if encodings:
                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(filename)
                print(f"تم ترميز وجه في الصورة {filename} بنجاح.")
            else:
                print(f"تحذير: لم يتم العثور على أي ترميز وجه في الصورة {filename}.")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "غير معروف"
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if face_distances.size > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
            face_names.append(name)
        face_locations = (np.array(face_locations) / self.frame_resizing).astype(int)
        return face_locations, face_names

class EmotionAnalyzer:
    def __init__(self):
        self.emotion_detector = FER(mtcnn=True)
        self.EMOTION_LIST = {
            'happy': 'سعيد',
            'sad': 'حزين',
            'angry': 'غاضب',
            'surprise': 'مندهش',
            'fear': 'خائف',
            'neutral': 'محايد'
        }

    def analyze_emotions(self, face_img):
        emotions = self.emotion_detector.detect_emotions(face_img)
        if emotions:
            dominant_emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
            return self.EMOTION_LIST.get(dominant_emotion, "غير معروف")
        return "غير معروف"

class AgeGenderAnalyzer:
    def __init__(self):
        self.age_net = cv2.dnn.readNetFromCaffe('data/deploy_age.prototxt', 'data/age_net.caffemodel')
        self.gender_net = cv2.dnn.readNetFromCaffe('data/deploy_gender.prototxt', 'data/gender_net.caffemodel')
        self.AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-24)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.GENDER_LIST = ['ذكر', 'أنثى']

    def analyze_age_gender(self, face_img):
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                       (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = self.AGE_BUCKETS[age_preds[0].argmax()]
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = self.GENDER_LIST[gender_preds[0].argmax()]
        return age, gender

class MoneyRecognizer:
    def __init__(self):
        self.model = YOLO('Trained_Models/best_15e.pt')
        self.money = {10: "5 جنيه", 1: "5 جنيه", 0: "10 جنيه", 3: "10 جنيه",
                      9: "20 جنيه", 5: "20 جنيه", 8: "50 جنيه",
                      2: "50 جنيه", 4: "100 جنيه", 11: "100 جنيه",
                      6: "200 جنيه", 7: "200 جنيه"}

    def recognize_money(self, frame):
        results = self.model.predict(source=frame, conf=0.4, verbose=False, save=False)
        total_money = 0
        money_positions = []
        if results and len(results[0].boxes.cls) > 0:
            for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                cls_index = int(cls.item())
                Currency = self.money.get(cls_index, "عملة غير معروفة")
                total_money += int(Currency.split()[0])
                money_positions.append((x1, Currency))
        return total_money, money_positions

class Assistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        pygame.mixer.init()
        self.face_recognizer = SimpleFacerec()
        self.face_recognizer.load_encoding_images("images/")
        self.emotion_analyzer = EmotionAnalyzer()
        self.age_gender_analyzer = AgeGenderAnalyzer()
        self.money_recognizer = MoneyRecognizer()
        self.cap = cv2.VideoCapture(0)
        self.face_recognition_active = False
        self.money_recognition_active = False
        self.stop_requested = False  # متغير للتحكم في إيقاف العمليات فور طلب المستخدم

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def speak(self, text):
        speak_text(text)

    def listen(self):
        """
        دالة الاستماع للأوامر الصوتية مع ضبط ضوضاء البيئة.
        تم تقليل وقت timeout إلى 2 ثانية و phrase_time_limit إلى 3 ثوان.
        كما يتم فحص كلمات التوقف بعد الحصول على الأمر.
        """
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("أستمع...")
            try:
                audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=3)
                command = self.recognizer.recognize_google(audio, language='ar-SA')
                print(f"الأمر: {command}")
                # تحليل صوتي إضافي: استخراج قيمة pitch باستخدام Par‑sel‑mouth لتحليل خصائص المتحدث
                voice_pitch = self.analyze_voice(audio)
                if voice_pitch:
                    print(f"متوسط نغمة الصوت: {voice_pitch:.2f} هرتز")
                return command
            except sr.UnknownValueError:
                print("لم أتمكن من فهم الصوت")
                return None
            except sr.RequestError as e:
                print(f"خطأ في الاتصال: {e}")
                return None
            except sr.WaitTimeoutError:
                print("لم يتم التقاط أي صوت في الوقت المحدد.")
                return None

    def analyze_voice(self, audio):
        """
        دالة لتحليل الصوت باستخدام مكتبة parselmouth واستخراج المتوسط الأساسي للنغمة (pitch)
        لتقدير خصائص المتحدث. تُحوّل بيانات AudioData إلى تنسيق WAV وتُستخدم parselmouth لاكتشاف النغمة.
        """
        try:
            # الحصول على بيانات WAV من audio (تعمل طريقة get_wav_data() في SpeechRecognition)
            wav_data = audio.get_wav_data()
            # إنشاء كائن Sound من parselmouth من البيانات
            sound = parselmouth.Sound(io.BytesIO(wav_data))
            pitch = sound.to_pitch()  # يقوم بتحويل الصوت إلى كائن Pitch
            mean_pitch = pitch.get_mean()  # حساب المتوسط في النغمة
            return mean_pitch
        except Exception as e:
            print(f"خطأ في تحليل الصوت: {e}")
        return None

    def get_time(self):
        now = datetime.datetime.now()
        hour = now.hour % 12 or 12
        minute = now.minute
        period = "صباحاً" if now.hour < 12 else "مساءً"
        return f"الساعة الآن {hour} و{minute} دقيقة {period}"

    def handle_time_command(self):
        current_time = self.get_time()
        print(current_time)
        self.speak(current_time)

    def handle_status_command(self):
        if self.face_recognition_active:
            status_message = "نظام التعرف على الوجه قيد التشغيل."
        elif self.money_recognition_active:
            status_message = "نظام التعرف على المال قيد التشغيل."
        else:
            status_message = "لا يوجد نظام قيد التشغيل."
        self.speak(status_message)

    def play_youtube_song(self, song_name):
        pywhatkit.playonyt(song_name)

    def play_media(self):
        self.speak("ما الذي تريد تشغيله؟")
        command = self.listen()
        if command:
            song_name = command
            self.speak('هل تريد تشغيل الأغنية وإيقاف النظام؟')
            confirmation = self.listen()
            if confirmation and 'نعم' in confirmation:
                self.speak(f'جاري تشغيل أغنية {song_name} على يوتيوب...')
                self.play_youtube_song(song_name)
                self.speak("إلى اللقاء، سيتم إغلاق النظام الآن.")
                sys.exit()
            else:
                self.speak(f'جاري تشغيل أغنية {song_name} على يوتيوب...')
                self.play_youtube_song(song_name)

    def introduce_self(self):
        introduction = (
            "مرحبا بكم، أنا نور، المساعد الذكي للمكفوفين وقد تم اختراعي في عام 2024. "
            "أنا هنا لمساعدة المكفوفين في العالم. "
            "أساعد في التعرف على الأشياء والتكيف مع البيئة المحيطة، قراءة الرسائل، "
            "وأخبارهم بالمواعيد، وأي شيء يطلبونه."
        )
        self.speak(introduction)

    def handle_knowledge_query(self, query):
        wikipedia_result = search_wikipedia(query)
        if wikipedia_result:
            self.speak(f"وجدت المعلومات التالية في ويكيبيديا: {wikipedia_result}")
        else:
            self.speak("عذرًا، لم أتمكن من العثور على إجابة لسؤالك.")

    def face_recognition_system(self):
        if self.face_recognition_active:
            return
        self.face_recognition_active = True
        self.speak("لقد قمت بطلب التعرف على الوجه. سيتم التعرف على الوجه الآن.")
        while not self.stop_requested:
            ret, frame = self.cap.read()
            if not ret:
                print("خطأ: فشل في التقاط صورة من الكاميرا.")
                break
            face_locations, face_names = self.face_recognizer.detect_known_faces(frame)
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                self.speak(f"مرحباً {name}")
                face_img = frame[top:bottom, left:right]
                age, gender = self.age_gender_analyzer.analyze_age_gender(face_img)
                self.speak(f"العمر: {age}, الجنس: {gender}")
                dominant_emotion = self.emotion_analyzer.analyze_emotions(face_img)
                self.speak(f"المشاعر: {dominant_emotion}")
            cv2.imshow("التعرف على الوجه", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.face_recognition_active = False
        self.stop_requested = False

    def money_recognition(self):
        if self.money_recognition_active:
            return
        self.money_recognition_active = True
        self.speak("لقد سمعت الأمر الصوتي 'شوف الفلوس'. سيتم التعرف على المال الآن.")
        while not self.stop_requested:
            ret, frame = self.cap.read()
            if not ret:
                print("خطأ: فشل في التقاط صورة من الكاميرا.")
                break
            total_money, money_positions = self.money_recognizer.recognize_money(frame)
            if total_money > 0:
                self.speak(f"إجمالي النقود: {total_money} جنيه مصري تم اكتشافها.")
                for _, currency in money_positions:
                    self.speak(currency)
            cv2.imshow('كشف العملات', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.money_recognition_active = False
        self.stop_requested = False

    def translate_system(self):
        self.speak("مرحبًا بك في نظام الترجمة. ما الأمر الصوتي الذي تريده؟")
        while True:
            command = self.listen()
            if command:
                if "عربي" in command or "بالعربي" in command or "بالمصري" in command:
                    self.speak("اخترت التحدث باللغة العربية. ابدأ التحدث الآن.")
                    arabic_text = self.listen()
                    if arabic_text:
                        english_translation = translator.translate(arabic_text, src='ar', dest='en').text
                        self.speak(f"الترجمة إلى الإنجليزية: {english_translation}")
                elif "انجليزي" in command or "بالانجليزي" in command:
                    self.speak("اخترت التحدث باللغة الإنجليزية. ابدأ التحدث الآن.")
                    english_text = self.listen()
                    if english_text:
                        arabic_translation = translator.translate(english_text, src='en', dest='ar').text
                        self.speak(f"الترجمة إلى العربية: {arabic_translation}")
                elif "توقف" in command:
                    self.speak("تم إيقاف نظام الترجمة. مع السلامة!")
                    break
                else:
                    self.speak("لم أفهم الأمر. الرجاء قول 'عربي' أو 'انجليزي' لبدء الترجمة، أو 'توقف' للإيقاف.")

    def run(self):
        # تعريف كلمات التوقف وكلمات إنهاء النظام
        stop_keywords = ["نور توقف", "شكراً يا نور", "شكراً"]
        terminate_keywords = ["اغلق النظام", "اقفل النظام"]

        self.speak("مرحبا بكم، أنا نور. كيف يمكنني مساعدتك؟")
        while True:
            command = self.listen()
            if command:
                # فحص كلمات التوقف: إذا وجدنا إحداها، يتم إيقاف العمليات الجارية وعرض سؤال متابعة للمستخدم.
                for keyword in stop_keywords:
                    if keyword in command:
                        self.speak("تم إيقاف الأمر الصوتي. هل تريد شيئًا آخر؟")
                        follow_up = self.listen()
                        if follow_up:
                            for term in terminate_keywords:
                                if term in follow_up:
                                    self.speak("مع السلامة، شكرًا لك!")
                                    sys.exit()
                        # عند استلام كلمة توقف نقوم بتجاهل الأمر الحالي.
                        command = ""
                        break

                if not command:
                    continue

                if any(phrase in command for phrase in ["نور قدمي نفسك", "نور اشرحي نفسك", "نور عرفينا بنفسك"]):
                    self.speak("حاضر سأقوم بتعريف نفسي لكم الآن.")
                    self.introduce_self()
                elif "مين واقف" in command:
                    threading.Thread(target=self.face_recognition_system, daemon=True).start()
                elif "شوف الفلوس" in command:
                    threading.Thread(target=self.money_recognition, daemon=True).start()
                elif "الوقت" in command or "كم الساعة" in command:
                    self.handle_time_command()
                elif "ما هي حالتي" in command:
                    self.handle_status_command()
                elif "شغلي" in command:
                    self.play_media()
                elif "عايز اعرف" in command:
                    query = command.split("عايز اعرف", 1)[1].strip()
                    if query:
                        self.speak(f"جاري البحث عن: {query}")
                        self.handle_knowledge_query(query)
                    else:
                        self.speak("لم أتمكن من سماع ما تريد معرفته.")
                elif "نور ترجمي" in command:
                    threading.Thread(target=self.translate_system, daemon=True).start()
                elif "خروج" in command:
                    self.speak("وداعاً!")
                    break
                # يمكن إضافة أوامر إضافية هنا...
            else:
                continue

if __name__ == "__main__":
    assistant = Assistant()
    assistant.run()
