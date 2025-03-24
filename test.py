import json
import os
import faiss
import numpy as np
import openai
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import httpx
from openai import OpenAI

# Настройка API ключа OpenAI
load_dotenv()  # Загружаем переменные окружения из .env файла
openai.api_key = os.getenv("OPENAI_API_KEY")  # Получаем ключ из переменных окружения

class VectorDBQuerier:
    def __init__(self, faiss_db_paths: List[str], model_name: str = "sentence-transformers/LaBSE"):
        """
        Инициализация класса для работы с векторными БД FAISS
        
        Args:
            faiss_db_paths: список путей к файлам .faiss
            model_name: название модели для эмбеддингов
        """
        self.faiss_dbs = []
        self.texts = []
        
        print(f"Загрузка модели {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        # Проверяем размерность эмбеддингов модели
        test_embedding = self.model.encode(["тестовый текст"])[0]
        print(f"Размерность эмбеддингов модели: {len(test_embedding)}")
        
        # Загрузка всех векторных БД
        for db_path in faiss_db_paths:
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"FAISS база данных не найдена: {db_path}")
            
            # Загрузка индекса FAISS
            index = faiss.read_index(db_path)
            print(f"Размерность индекса FAISS в {db_path}: {index.d}")
            
            self.faiss_dbs.append(index)
            
            # Получаем имя файла без пути и расширения
            db_filename = os.path.basename(db_path)
            db_name = os.path.splitext(db_filename)[0]
            
            # Формируем путь к соответствующему JSON-файлу
            text_path = os.path.join("data/test_set/databases/chunked_reports", f"{db_name}.json")
            
            if not os.path.exists(text_path):
                raise FileNotFoundError(f"Файл с текстами не найден: {text_path}")
            
            # Загрузка текстов
            with open(text_path, 'r', encoding='utf-8') as f:
                texts = json.load(f)
            self.texts.append(texts)
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Кодирование текста с расширением эмбеддинга до нужной размерности
        
        Args:
            text: текст для кодирования
            
        Returns:
            numpy array размерности 3072
        """
        # Получаем базовый эмбеддинг размерности 768
        base_embedding = self.model.encode([text])[0]
        
        # Расширяем эмбеддинг до размерности 3072 путем повторения 4 раза
        expanded_embedding = np.tile(base_embedding, 4)
        
        return expanded_embedding

    def preprocess_query(self, query: str) -> str:
        """
        Предварительная обработка запроса через GPT для улучшения качества поиска
        
        Args:
            query: исходный запрос пользователя
            
        Returns:
            улучшенный запрос для векторного поиска
        """
        # Настройка прокси для запроса
        proxy = "http://user156811:eb49hn@45.159.182.77:5442"
        
        # Создание клиента OpenAI с прокси
        client = OpenAI(
            api_key=openai.api_key,
            http_client=httpx.Client(proxies=proxy)
        )
        
        prompt = f"""Переформулируй запрос для улучшения векторного поиска по учебным материалам. 
Добавь ключевые термины и синонимы, которые могут быть в тексте.
Сохрани основной смысл запроса, но сделай его более полным для поиска.

Исходный запрос: {query}

Дай только переформулированный запрос, без пояснений."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Ты - эксперт по улучшению запросов для векторного поиска в технической документации."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=256
        )
        
        enhanced_query = response.choices[0].message.content.strip()
        print(f"Исходный запрос: {query}")
        print(f"Улучшенный запрос: {enhanced_query}")
        return enhanced_query

    def analyze_chunks_relevance(self, query: str, chunks: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Анализ релевантности найденных чанков через GPT
        
        Args:
            query: исходный запрос пользователя
            chunks: найденные фрагменты текста
            
        Returns:
            tuple: (достаточно ли информации, рекомендации по расширению поиска)
        """
        # Подготовка контекста из чанков
        context = "\n\n".join([f"[Фрагмент {i+1}]: {chunk['text']}" for i, chunk in enumerate(chunks)])
        
        prompt = f"""Проанализируй, содержат ли предоставленные фрагменты текста достаточно информации для ответа на вопрос.

Вопрос: {query}

Фрагменты текста:
{context}

Дай ответ в формате:
1. Достаточно ли информации (да/нет)
2. Если информации недостаточно, укажи какие дополнительные темы или термины нужно искать"""

        # Настройка прокси для запроса
        proxy = "http://user156811:eb49hn@45.159.182.77:5442"
        
        # Создание клиента OpenAI с прокси
        client = OpenAI(
            api_key=openai.api_key,
            http_client=httpx.Client(proxies=proxy)
        )
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Ты - эксперт по анализу технической документации. Твоя задача - определить, достаточно ли информации в предоставленных фрагментах для ответа на вопрос."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=256
        )
        
        analysis = response.choices[0].message.content.strip().lower()
        
        # Определяем, достаточно ли информации
        has_enough_info = "да" in analysis.split('\n')[0]
        suggestions = analysis.split('\n', 1)[1] if '\n' in analysis else ""
        
        return has_enough_info, suggestions

    def search(self, query: str, top_k: int = 5, max_attempts: int = 3) -> List[Dict[str, Any]]:
        """
        Поиск релевантных фрагментов текста по запросу с возможностью расширения поиска
        
        Args:
            query: запрос пользователя
            top_k: количество возвращаемых результатов
            max_attempts: максимальное количество попыток поиска
            
        Returns:
            список найденных фрагментов
        """
        current_query = query
        all_results = []
        attempts = 0
        
        while attempts < max_attempts:
            # Предварительная обработка запроса
            enhanced_query = self.preprocess_query(current_query)
            
            # Поиск по текущему запросу
            query_embedding = self.encode_text(enhanced_query)
            current_results = []
            
            for i, index in enumerate(self.faiss_dbs):
                query_embedding_resized = np.array([query_embedding], dtype=np.float32)
                distances, indices = index.search(query_embedding_resized, top_k)
                
                chunks = self.texts[i]['content']['chunks']
                
                for j, idx in enumerate(indices[0]):
                    if idx != -1 and idx < len(chunks):
                        try:
                            chunk = chunks[idx]
                            result = {
                                "text": chunk["text"],
                                "distance": float(distances[0][j]),
                                "db_index": i,
                                "text_index": idx,
                                "page": chunk["page"]
                            }
                            current_results.append(result)
                        except Exception as e:
                            print(f"Ошибка при обработке результата: {e}")
                            continue
            
            # Добавляем новые результаты к общему списку
            all_results.extend(current_results)
            
            # Сортируем все результаты по релевантности
            all_results.sort(key=lambda x: x["distance"])
            unique_results = []
            seen_texts = set()
            
            # Удаляем дубликаты
            for result in all_results:
                if result["text"] not in seen_texts:
                    seen_texts.add(result["text"])
                    unique_results.append(result)
            
            # Проверяем релевантность через GPT
            has_enough_info, suggestions = self.analyze_chunks_relevance(query, unique_results[:top_k])
            
            if has_enough_info:
                print("Найдена достаточная информация для ответа")
                return unique_results[:top_k]
            
            # Если информации недостаточно и есть ещё попытки
            if attempts < max_attempts - 1:
                print(f"Попытка {attempts + 1}: Информации недостаточно. Расширяем поиск.")
                print(f"Рекомендации по расширению: {suggestions}")
                current_query = suggestions if suggestions else query
            
            attempts += 1
        
        print("Достигнуто максимальное количество попыток поиска")
        return unique_results[:top_k]

def generate_answer(query: str, context: List[Dict[str, Any]]) -> str:
    """
    Генерация ответа с использованием ChatGPT
    
    Args:
        query: вопрос пользователя
        context: контекст из векторной БД
        
    Returns:
        ответ на вопрос
    """
    # Формируем контекст из текстов, добавляя информацию о странице
    context_texts = [f"[Страница {item['page']}] {item['text']}" for item in context]
    context_text = "\n\n".join(context_texts)
    
    prompt = f"""На основе следующей информации из методических материалов ответь на вопрос.
    
Информация из методических материалов:
{context_text}

Вопрос: {query}

Дай подробный и точный ответ, основываясь только на предоставленной информации. Если информации недостаточно, укажи это."""

    proxy = "http://user156811:eb49hn@45.159.182.77:5442"
    
    client = OpenAI(
        api_key=openai.api_key,
        http_client=httpx.Client(proxies=proxy)
    )
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Ты - ассистент, который помогает студентам с вопросами по учебным материалам."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
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
            "data/test_set/databases/vector_dbs/74332 (1).faiss"
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
        import traceback
        error_traceback = traceback.format_exc()
        return {"error": f"Произошла ошибка: {str(e)}", "traceback": error_traceback}

# Пример использования
if __name__ == "__main__":
    # Пример JSON-запроса
    sample_query = '{"question": "Напиши формулу коэффициента усиления по мощности в схеме каскада с общим эмиттером"}'
    
    # Обработка запроса
    result = process_json_query(sample_query)
    
    # Вывод результата
    print(json.dumps(result, ensure_ascii=False, indent=2))
