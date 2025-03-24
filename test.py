import os
import time
import glob
from pathlib import Path
from docling.document_converter import DocumentConverter
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance
import io
import numpy as np
import concurrent.futures

def check_gpu():
    """Проверка доступности GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU доступен: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("GPU не доступен, будет использован CPU")
            return False
    except ImportError:
        print("PyTorch не установлен, будет использован CPU")
        return False

def preprocess_pdf(pdf_path, output_path=None, enhance=True, dpi=300):
    """Предварительная обработка PDF для улучшения качества распознавания"""
    if output_path is None:
        output_path = os.path.splitext(pdf_path)[0] + "_preprocessed.pdf"
    
    print(f"Предварительная обработка PDF {pdf_path}...")
    
    # Открываем исходный PDF
    doc = fitz.open(pdf_path)
    # Создаем новый PDF для обработанных страниц
    new_doc = fitz.open()
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Получаем изображение страницы с высоким разрешением
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        img_data = pix.samples
        
        # Преобразуем в изображение PIL для обработки
        img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
        
        if enhance:
            # Улучшение контрастности
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)  # Увеличиваем контраст
            
            # Улучшение резкости
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.5)  # Увеличиваем резкость
        
        # Преобразуем обратно в формат, подходящий для PDF
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        # Создаем новую страницу с улучшенным изображением
        new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, stream=img_bytes)
    
    # Сохраняем обработанный PDF
    new_doc.save(output_path)
    new_doc.close()
    doc.close()
    
    print(f"Предварительно обработанный PDF сохранен в {output_path}")
    return output_path

def process_pdf(pdf_path, output_txt_path=None, output_md_path=None, output_json_path=None, preprocess=False):
    """Обработка PDF с использованием Docling"""
    if not os.path.exists(pdf_path):
        print(f"Файл {pdf_path} не найден!")
        return None
    
    # Если выходные пути не указаны, создаем их на основе имени PDF файла
    if output_txt_path is None:
        output_txt_path = os.path.splitext(pdf_path)[0] + "_docling.txt"
    
    if output_md_path is None:
        output_md_path = os.path.splitext(pdf_path)[0] + "_docling.md"
        
    if output_json_path is None:
        output_json_path = os.path.splitext(pdf_path)[0] + "_docling.json"
    
    start_time = time.time()
    
    # Проверка GPU
    use_gpu = check_gpu()
    
    # Предварительная обработка PDF при необходимости
    processed_pdf = pdf_path
    if preprocess:
        processed_pdf = preprocess_pdf(pdf_path)
    
    print(f"Обработка документа {processed_pdf}...")
    
    # Создаем конвертер документов с дополнительными параметрами
    converter = DocumentConverter(
        # Можно настроить дополнительные параметры для улучшения распознавания
        ocr_languages=["rus", "eng"],  # Языки для OCR
        ocr_force=True,  # Принудительное использование OCR даже для PDF с текстовым слоем
        ocr_dpi=300,     # Повышенное разрешение для OCR
    )
    
    # Конвертируем PDF в структурированный документ
    try:
        result = converter.convert(processed_pdf)
        
        # Получаем документ
        document = result.document
        
        # Сохраняем результат в текстовом формате
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(document.export_to_text())
        
        # Сохраняем результат в формате Markdown (сохраняет таблицы в формате markdown)
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write(document.export_to_markdown())
            
        # Сохраняем результат в формате JSON для дальнейшей обработки
        with open(output_json_path, 'w', encoding='utf-8') as f:
            f.write(document.export_to_json())
        
        print(f"Обработка завершена за {time.time() - start_time:.2f} секунд.")
        print(f"Текстовый результат сохранен в {output_txt_path}")
        print(f"Markdown результат сохранен в {output_md_path}")
        print(f"JSON результат сохранен в {output_json_path}")
        
        # Удаляем временный файл, если была предварительная обработка
        if preprocess and processed_pdf != pdf_path and os.path.exists(processed_pdf):
            os.remove(processed_pdf)
        
        return document
    
    except Exception as e:
        print(f"Ошибка при обработке {pdf_path}: {str(e)}")
        
        # Удаляем временный файл, если была предварительная обработка
        if preprocess and processed_pdf != pdf_path and os.path.exists(processed_pdf):
            os.remove(processed_pdf)
            
        return None

def process_directory(directory_path, output_dir=None, preprocess=True, parallel=True, max_workers=4):
    """Обработка всех PDF файлов в указанной директории"""
    if not os.path.exists(directory_path):
        print(f"Директория {directory_path} не найдена!")
        return
    
    if output_dir is None:
        output_dir = os.path.join(directory_path, "processed")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем список всех PDF файлов в директории
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
    
    if not pdf_files:
        print(f"В директории {directory_path} не найдено PDF файлов.")
        return
    
    print(f"Найдено {len(pdf_files)} PDF файлов для обработки.")
    
    if parallel and len(pdf_files) > 1:
        # Параллельная обработка файлов
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for pdf_file in pdf_files:
                file_name = os.path.basename(pdf_file)
                output_txt = os.path.join(output_dir, os.path.splitext(file_name)[0] + "_docling.txt")
                output_md = os.path.join(output_dir, os.path.splitext(file_name)[0] + "_docling.md")
                output_json = os.path.join(output_dir, os.path.splitext(file_name)[0] + "_docling.json")
                
                futures.append(
                    executor.submit(
                        process_pdf, 
                        pdf_file, 
                        output_txt, 
                        output_md,
                        output_json,
                        preprocess
                    )
                )
            
            # Ожидаем завершения всех задач
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Ошибка при параллельной обработке: {str(e)}")
    else:
        # Последовательная обработка файлов
        for pdf_file in pdf_files:
            file_name = os.path.basename(pdf_file)
            output_txt = os.path.join(output_dir, os.path.splitext(file_name)[0] + "_docling.txt")
            output_md = os.path.join(output_dir, os.path.splitext(file_name)[0] + "_docling.md")
            output_json = os.path.join(output_dir, os.path.splitext(file_name)[0] + "_docling.json")
            
            process_pdf(pdf_file, output_txt, output_md, output_json, preprocess)
    
    print(f"Обработка всех PDF файлов завершена. Результаты сохранены в {output_dir}")

if __name__ == "__main__":
    # Указываем пути к файлам прямо в коде
    # Для обработки одного файла:
    # pdf_path = "путь_к_методичке.pdf"  # Укажите здесь путь к вашему PDF файлу
    # output_txt_path = "результат_docling.txt"  # Укажите здесь путь для сохранения текстового результата
    # output_md_path = "результат_docling.md"  # Укажите здесь путь для сохранения markdown результата
    # output_json_path = "результат_docling.json"  # Укажите здесь путь для сохранения json результата
    # process_pdf(pdf_path, output_txt_path, output_md_path, output_json_path, preprocess=True)
    
    # Для обработки всех файлов в директории:
    directory_path = "путь_к_директории_с_методичками"  # Укажите здесь путь к директории с PDF файлами
    output_dir = "путь_к_директории_для_результатов"  # Укажите здесь путь для сохранения результатов
    
    # Параметры обработки:
    preprocess = True  # Предварительная обработка PDF для улучшения качества распознавания
    parallel = True  # Параллельная обработка файлов
    max_workers = 4  # Максимальное количество параллельных потоков
    
    process_directory(directory_path, output_dir, preprocess, parallel, max_workers)
