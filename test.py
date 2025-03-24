import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io
import os

def preprocess_image(img):
    """Улучшает качество изображения перед OCR"""
    # Конвертируем в OpenCV-формат (numpy array)
    img_cv = np.array(img)
    
    # Улучшаем контраст (CLAHE - адаптивное выравнивание гистограммы)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    # Бинаризация (если нужно)
    gray = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2GRAY)
    _, img_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return Image.fromarray(img_bin)

def pdf_to_high_accuracy_ocr(pdf_path, output_txt_path=None, dpi=400, lang="rus+eng"):
    """
    Извлекает текст из PDF с максимальной точностью OCR.
    
    :param pdf_path: Путь к PDF
    :param output_txt_path: Куда сохранить текст (если None — возвращает строку)
    :param dpi: Разрешение сканирования (рекомендуется 300-600)
    :param lang: Языки для Tesseract (например, "rus+eng")
    :return: Текст или None (если сохранено в файл)
    """
    doc = fitz.open(pdf_path)
    full_text = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Рендерим страницу как изображение с высоким DPI
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("ppm")
        img_pil = Image.open(io.BytesIO(img_bytes))
        
        # Улучшаем изображение перед OCR
        processed_img = preprocess_image(img_pil)
        
        # Настройки Tesseract для максимальной точности
        custom_config = r"""
            --oem 3  # Используем LSTM-движок
            --psm 6   # Авто-определение структуры текста
            -c preserve_interword_spaces=1
            -c tessedit_char_whitelist=.,!?:;-—()«»№@%s
        """
        
        # Применяем OCR
        page_text = pytesseract.image_to_string(
            processed_img,
            lang=lang,
            config=custom_config
        )
        
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
    input_pdf = "document.pdf"
    output_txt = "output_ocr.txt"
    
    if not os.path.exists(input_pdf):
        print(f"❌ Файл {input_pdf} не найден!")
    else:
        pdf_to_high_accuracy_ocr(input_pdf, output_txt)
        print("✅ OCR завершён с максимальной точностью!")