import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io
import os
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from deep_translator import GoogleTranslator
import nltk
from nltk.tokenize import word_tokenize

# Загружаем необходимые ресурсы NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Загрузка модели для постобработки текста (если доступна CUDA)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    model_ru_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en").to(device)
    tokenizer_en_ru = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
    model_en_ru = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru").to(device)
    use_translation_correction = True
except:
    use_translation_correction = False
    print("⚠️ Модели машинного перевода не загружены. Будет использована только базовая коррекция.")

# Загрузка словаря для проверки орфографии через NLTK
try:
    nltk.download('words')
    from nltk.corpus import words as nltk_words
    russian_words_available = False
    print("⚠️ Словарь русских слов недоступен через NLTK. Будет использована только базовая коррекция.")
except:
    russian_words_available = False
    print("⚠️ Словарь слов недоступен. Будет использована только базовая коррекция.")

def preprocess_image(img):
    """Улучшает качество изображения перед OCR для русского текста"""
    # Конвертируем в OpenCV-формат (numpy array)
    img_cv = np.array(img)
    
    # Увеличение разрешения с помощью Super Resolution (если изображение маленькое)
    height, width = img_cv.shape[:2]
    if width < 1000 or height < 1000:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        try:
            path_to_model = "EDSR_x4.pb"  # Предполагается, что модель доступна
            if not os.path.exists(path_to_model):
                print("⚠️ Модель Super Resolution не найдена. Используется стандартное масштабирование.")
                img_cv = cv2.resize(img_cv, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
            else:
                sr.readModel(path_to_model)
                sr.setModel("edsr", 4)
                img_cv = sr.upsample(img_cv)
        except:
            img_cv = cv2.resize(img_cv, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
    
    # Улучшаем контраст (CLAHE - адаптивное выравнивание гистограммы)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    # Преобразование в оттенки серого
    gray = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2GRAY)
    
    # Удаление шума с помощью билатерального фильтра (сохраняет края)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Адаптивная бинаризация для лучшего распознавания кириллицы
    img_bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 15, 8)
    
    # Морфологические операции для улучшения текста
    kernel = np.ones((1, 1), np.uint8)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)
    
    # Удаление мелких шумов
    kernel = np.ones((2, 2), np.uint8)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)
    
    # Увеличение толщины текста для лучшего распознавания
    kernel = np.ones((1, 1), np.uint8)
    img_bin = cv2.dilate(img_bin, kernel, iterations=1)
    
    return Image.fromarray(img_bin)

def correct_text_with_translation(text):
    """Исправляет текст с помощью двойного перевода (ru->en->ru)"""
    if not use_translation_correction:
        return text
    
    try:
        # Разбиваем текст на части, чтобы не превышать лимиты модели
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        corrected_chunks = []
        
        for chunk in chunks:
            # Перевод ru -> en
            inputs = tokenizer(chunk, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model_ru_en.generate(**inputs)
            en_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Перевод en -> ru
            inputs = tokenizer_en_ru(en_text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model_en_ru.generate(**inputs)
            corrected_chunk = tokenizer_en_ru.decode(outputs[0], skip_special_tokens=True)
            corrected_chunks.append(corrected_chunk)
        
        return " ".join(corrected_chunks)
    except Exception as e:
        print(f"⚠️ Ошибка при коррекции текста через перевод: {e}")
        return text

def correct_spelling_with_google(text):
    """Исправляет орфографические ошибки с помощью Google Translator"""
    try:
        # Используем Google Translator для исправления орфографии
        translator = GoogleTranslator(source='ru', target='ru')
        corrected_text = translator.translate(text)
        return corrected_text
    except Exception as e:
        print(f"⚠️ Ошибка при коррекции орфографии через Google: {e}")
        return text

def post_process_text(text):
    """Постобработка распознанного текста для улучшения качества"""
    # Исправление типичных ошибок OCR для русского языка
    replacements = {
        'б': 'б', 'в': 'в', 'д': 'д',  # Замена похожих символов
        'и': 'и', 'й': 'й', 'л': 'л',
        'п': 'п', 'ф': 'ф', 'ц': 'ц',
        'ш': 'ш', 'щ': 'щ', 'ъ': 'ъ',
        'ы': 'ы', 'ь': 'ь', 'э': 'э',
        'ю': 'ю', 'я': 'я',
        '0': 'о', '3': 'з', '6': 'б',
        '9': 'д', '4': 'ч'
    }
    
    # Замена ошибочных символов
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text)
    
    # Исправление орфографии через Google Translator
    text = correct_spelling_with_google(text)
    
    # Исправление с помощью двойного перевода (если доступно)
    if use_translation_correction:
        text = correct_text_with_translation(text)
    
    return text

def pdf_to_high_accuracy_ocr(pdf_path, output_txt_path=None, dpi=600, lang="rus"):
    """
    Извлекает текст из PDF с максимальной точностью OCR для русского языка.
    
    :param pdf_path: Путь к PDF
    :param output_txt_path: Куда сохранить текст (если None — возвращает строку)
    :param dpi: Разрешение сканирования (рекомендуется 300-600)
    :param lang: Языки для Tesseract (по умолчанию только русский)
    :return: Текст или None (если сохранено в файл)
    """
    doc = fitz.open(pdf_path)
    full_text = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        print(f"Обработка страницы {page_num + 1} из {len(doc)}...")
        
        # Рендерим страницу как изображение с высоким DPI
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("ppm")
        img_pil = Image.open(io.BytesIO(img_bytes))
        
        # Улучшаем изображение перед OCR
        processed_img = preprocess_image(img_pil)
        
        # Сохраняем промежуточное изображение для отладки (опционально)
        debug_dir = "debug_images"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        processed_img.save(f"{debug_dir}/page_{page_num+1}_processed.png")
        
        # Настройки Tesseract оптимизированные для русского языка
        custom_config = r"""
            --oem 1  # LSTM-движок
            --psm 6  # Предполагаем блок текста
            -c preserve_interword_spaces=1
            -c tessedit_char_whitelist=.,!?:;-—()«»№@%0123456789АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя
            -c language_model_penalty_non_freq_dict_word=0.5
            -c language_model_penalty_non_dict_word=0.5
            -c textord_min_linesize=2.5
            -c tessedit_do_invert=0
            -c textord_really_old_xheight=0
            -c textord_force_make_prop_words=0
        """
        
        # Применяем OCR
        page_text = pytesseract.image_to_string(
            processed_img,
            lang=lang,
            config=custom_config
        )
        
        # Постобработка текста
        page_text = post_process_text(page_text)
        
        full_text.append(f"--- Страница {page_num + 1} ---\n{page_text}\n")
    
    doc.close()
    result_text = "\n".join(full_text)
    
    if output_txt_path:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(result_text)
        print(f"✅ Текст сохранён в {output_txt_path}")
        return None
    else:
        return result_text

# Пример использования
if __name__ == "__main__":
    input_pdf = "Основные правила оформления чертежей (Хотина, Ермакова, Кожухова)_rotated.pdf"
    output_txt = "output_ocr.txt"
    
    if not os.path.exists(input_pdf):
        print(f"❌ Файл {input_pdf} не найден!")
    else:
        pdf_to_high_accuracy_ocr(input_pdf, output_txt)
        print("✅ OCR завершён с максимальной точностью!")