import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os

def pdf_to_text_with_ocr(pdf_path, output_txt_path=None, dpi=300):
    """
    Извлекает текст из PDF с использованием OCR.
    
    :param pdf_path: Путь к PDF файлу
    :param output_txt_path: Путь для сохранения текста (если None, возвращает текст как строку)
    :param dpi: Разрешение для рендеринга изображений (по умолчанию 300 DPI)
    :return: Текст или None, если output_txt_path указан
    """
    # Открываем PDF файл
    doc = fitz.open(pdf_path)
    text = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Получаем изображение страницы с высоким разрешением
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("ppm")  # Конвертируем в PPM формат
        
        # Открываем изображение с помощью PIL
        img = Image.open(io.BytesIO(img_bytes))
        
        # Применяем OCR к изображению
        page_text = pytesseract.image_to_string(img, lang='rus+eng')  # Для русского и английского
        
        text += f"--- Страница {page_num + 1} ---\n{page_text}\n"
    
    doc.close()
    
    if output_txt_path:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Текст сохранен в: {output_txt_path}")
        return None
    else:
        return text

# Пример использования
if __name__ == "__main__":
    # Укажите путь к вашему PDF файлу
    input_pdf = "Основные правила оформления чертежей (Хотина, Ермакова, Кожухова)_rotated.pdf"
    output_txt = "output.txt"
    
    # Проверяем существование файла
    if not os.path.exists(input_pdf):
        print(f"Ошибка: файл {input_pdf} не найден!")
    else:
        # Извлекаем текст
        pdf_to_text_with_ocr(input_pdf, output_txt)
        print("OCR завершен!")