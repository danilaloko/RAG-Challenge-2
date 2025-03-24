import json
import os
import faiss
import numpy as np
import openai
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import httpx
from openai import OpenAI

# Настройка API ключа OpenAI
load_dotenv()  # Загружаем переменные окружения из .env файла
openai.api_key = os.getenv("OPENAI_API_KEY")  # Получаем ключ из переменных окружения

class VectorDBQuerier:
    def __init__(self, faiss_db_paths: List[str], model_name: str = "all-MiniLM-L6-v2"):
        """
        Инициализация класса для работы с векторными БД FAISS
        
        Args:
            faiss_db_paths: список путей к файлам .faiss
            model_name: название модели для эмбеддингов
        """
        self.faiss_dbs = []
        self.texts = []
        self.model = SentenceTransformer(model_name)
        
        # Загрузка всех векторных БД
        for db_path in faiss_db_paths:
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"FAISS база данных не найдена: {db_path}")
            
            # Предполагаем, что для каждой .faiss БД есть соответствующий .json файл с текстами
            text_path = db_path.replace('.faiss', '.json')
            if not os.path.exists(text_path):
                raise FileNotFoundError(f"Файл с текстами не найден: {text_path}")
            
            # Загрузка индекса FAISS
            index = faiss.read_index(db_path)
            self.faiss_dbs.append(index)
            
            # Загрузка текстов
            with open(text_path, 'r', encoding='utf-8') as f:
                texts = json.load(f)
            self.texts.append(texts)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Поиск релевантных фрагментов текста по запросу
        
        Args:
            query: текстовый запрос
            top_k: количество возвращаемых результатов
            
        Returns:
            список словарей с найденными фрагментами и их метаданными
        """
        # Получение эмбеддинга запроса
        query_embedding = self.model.encode([query])[0]
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        all_results = []
        
        # Поиск по всем базам данных
        for i, index in enumerate(self.faiss_dbs):
            distances, indices = index.search(query_embedding, top_k)
            
            # Формирование результатов
            for j, idx in enumerate(indices[0]):
                if idx != -1:  # Проверка на валидный индекс
                    result = {
                        "text": self.texts[i][idx],
                        "distance": float(distances[0][j]),
                        "db_index": i,
                        "text_index": idx
                    }
                    all_results.append(result)
        
        # Сортировка результатов по релевантности (меньшее расстояние = более релевантно)
        all_results.sort(key=lambda x: x["distance"])
        
        return all_results[:top_k]

def generate_answer(query: str, context: List[Dict[str, Any]]) -> str:
    """
    Генерация ответа с использованием ChatGPT
    
    Args:
        query: вопрос пользователя
        context: контекст из векторной БД
        
    Returns:
        ответ на вопрос
    """
    # Формирование контекста для запроса к ChatGPT
    context_text = "\n\n".join([item["text"] for item in context])
    
    # Формирование промпта
    prompt = f"""На основе следующей информации из методических материалов ответь на вопрос.
    
Информация из методических материалов:
{context_text}

Вопрос: {query}

Дай подробный и точный ответ, основываясь только на предоставленной информации. Если информации недостаточно, укажи это."""

    # Настройка прокси для запроса
    proxy = "http://user156811:eb49hn@45.159.182.77:5442"
    
    # Создание клиента OpenAI с прокси
    client = OpenAI(
        api_key=openai.api_key,
        http_client=httpx.Client(proxies=proxy)
    )
    
    # Запрос к ChatGPT
    response = client.chat.completions.create(
        model="gpt-4o",  # или "gpt-4" для более качественных ответов
        messages=[
            {"role": "system", "content": "Ты - ассистент, который помогает студентам с вопросами по учебным материалам."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Низкая температура для более точных ответов
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def process_json_query(json_query: str) -> Dict[str, Any]:
    """
    Обработка запроса в формате JSON
    
    Args:
        json_query: строка с JSON-запросом
        
    Returns:
        словарь с ответом
    """
    try:
        # Парсинг JSON
        query_data = json.loads(json_query)
        
        # Проверка наличия поля с вопросом
        if "question" not in query_data:
            return {"error": "В запросе отсутствует поле 'question'"}
        
        question = query_data["question"]
        
        # Пути к векторным БД (замените на свои)
        faiss_db_paths = [
            "data/test_set/databases/chunked_reports/74332 (1).faiss"
            # Добавьте другие пути при необходимости
        ]
        
        # Создание объекта для работы с векторными БД
        querier = VectorDBQuerier(faiss_db_paths)
        
        # Поиск релевантных фрагментов
        search_results = querier.search(question)
        
        # Генерация ответа
        answer = generate_answer(question, search_results)
        
        # Формирование ответа
        response = {
            "question": question,
            "answer": answer,
            "sources": [{"text": item["text"], "relevance": 1.0 - item["distance"]} for item in search_results]
        }
        
        return response
    
    except json.JSONDecodeError:
        return {"error": "Некорректный формат JSON"}
    except Exception as e:
        return {"error": f"Произошла ошибка: {str(e)}"}

# Пример использования
if __name__ == "__main__":
    # Пример JSON-запроса
    sample_query = '{"question": "Напиши формулу коэффициента усиления по мощности в схеме каскада с общим эмиттером"}'
    
    # Обработка запроса
    result = process_json_query(sample_query)
    
    # Вывод результата
    print(json.dumps(result, ensure_ascii=False, indent=2))
