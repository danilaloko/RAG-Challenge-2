import os
import time
import torch
import easyocr
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io

def check_gpu():
    """Проверка доступности GPU"""
    if torch.cuda.is_available():
        print(f"GPU доступен: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("GPU не доступен, будет использован CPU")
        return False

def extract_pdf_images(pdf_path, dpi=300):
    """Извлечение изображений из PDF файла с указанным DPI"""
    images = []
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        img_data = pix.samples
        img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
        images.append(np.array(img))
        
    return images

def perform_ocr(images, use_gpu=True):
    """Выполнение OCR с использованием EasyOCR"""
    # Используем русский и английский языки для распознавания
    languages = ['ru', 'en']
    
    # Инициализация ридера
    reader = easyocr.Reader(languages, gpu=use_gpu)
    
    all_text = []
    for i, img in enumerate(images):
        print(f"Обработка страницы {i+1}/{len(images)}...")
        result = reader.readtext(img)
        
        # Извлечение текста из результатов
        page_text = []
        for detection in result:
            text = detection[1]
            page_text.append(text)
        
        all_text.append(' '.join(page_text))
    
    return all_text

def main(pdf_path, output_txt_path=None):
    """Основная функция для выполнения OCR над PDF"""
    if not os.path.exists(pdf_path):
        print(f"Файл {pdf_path} не найден!")
        return
    
    # Если выходной путь не указан, создаем его на основе имени PDF файла
    if output_txt_path is None:
        output_txt_path = os.path.splitext(pdf_path)[0] + "_ocr.txt"
    
    start_time = time.time()
    
    # Проверка GPU
    use_gpu = check_gpu()
    
    # Извлечение изображений из PDF
    print("Извлечение изображений из PDF...")
    images = extract_pdf_images(pdf_path)
    
    # Выполнение OCR
    print("Выполнение OCR...")
    extracted_text = perform_ocr(images, use_gpu)
    
    # Сохранение результатов
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n\n--- СТРАНИЦА РАЗДЕЛИТЕЛЬ ---\n\n'.join(extracted_text))
    
    print(f"OCR завершен за {time.time() - start_time:.2f} секунд.")
    print(f"Результат сохранен в {output_txt_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OCR для PDF с использованием GPU')
    parser.add_argument('Основные правила оформления чертежей (Хотина, Ермакова, Кожухова)_rotated.pdf', help='Путь к PDF файлу')
    parser.add_argument('--output', help='Путь для сохранения результата (по умолчанию: имя_pdf_ocr.txt)')
    
    args = parser.parse_args()
    
    main(args.pdf_path, args.output)
