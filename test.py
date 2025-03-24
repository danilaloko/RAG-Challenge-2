import os
import time
from docling.document_converter import DocumentConverter

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

def process_pdf(pdf_path, output_txt_path=None, output_md_path=None):
    """Обработка PDF с использованием Docling"""
    if not os.path.exists(pdf_path):
        print(f"Файл {pdf_path} не найден!")
        return
    
    # Если выходные пути не указаны, создаем их на основе имени PDF файла
    if output_txt_path is None:
        output_txt_path = os.path.splitext(pdf_path)[0] + "_docling.txt"
    
    if output_md_path is None:
        output_md_path = os.path.splitext(pdf_path)[0] + "_docling.md"
    
    start_time = time.time()
    
    # Проверка GPU
    use_gpu = check_gpu()
    
    print(f"Обработка документа {pdf_path}...")
    
    # Создаем конвертер документов
    converter = DocumentConverter()
    
    # Конвертируем PDF в структурированный документ
    # Docling автоматически использует GPU, если он доступен
    result = converter.convert(pdf_path)
    
    # Получаем документ
    document = result.document
    
    # Сохраняем результат в текстовом формате
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(document.export_to_text())
    
    # Сохраняем результат в формате Markdown (сохраняет таблицы в формате markdown)
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(document.export_to_markdown())
    
    # Можно также сохранить в JSON для дальнейшей обработки
    # with open(output_json_path, 'w', encoding='utf-8') as f:
    #     f.write(document.export_to_json())
    
    print(f"Обработка завершена за {time.time() - start_time:.2f} секунд.")
    print(f"Текстовый результат сохранен в {output_txt_path}")
    print(f"Markdown результат сохранен в {output_md_path}")
    
    return document

if __name__ == "__main__":
    # Указываем пути к файлам прямо в коде
    pdf_path = "Основные правила оформления чертежей (Хотина, Ермакова, Кожухова)_rotated.pdf"  # Укажите здесь путь к вашему PDF файлу
    output_txt_path = "результат_docling.txt"  # Укажите здесь путь для сохранения текстового результата
    output_md_path = "результат_docling.md"  # Укажите здесь путь для сохранения markdown результата
    
    process_pdf(pdf_path, output_txt_path, output_md_path)
