import json
from typing import Union, Dict, List, Optional
import re
from pathlib import Path
from src.retrieval import VectorRetriever, HybridRetriever
from src.api_requests import APIProcessor
from tqdm import tqdm
import pandas as pd
import threading
import concurrent.futures


class QuestionsProcessor:
    def __init__(
        self,
        vector_db_dir: Union[str, Path] = './vector_dbs',
        documents_dir: Union[str, Path] = './documents',
        questions_file_path: Optional[Union[str, Path]] = None,
        subset_path: Optional[Union[str, Path]] = None,
        parent_document_retrieval: bool = False,
        llm_reranking: bool = False,
        llm_reranking_sample_size: int = 20,
        top_n_retrieval: int = 10,
        parallel_requests: int = 10,
        api_provider: str = "openai",
        answering_model: str = "gpt-4o-2024-08-06",
        full_context: bool = False
    ):
        self.questions = self._load_questions(questions_file_path)
        self.documents_dir = Path(documents_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.subset_path = Path(subset_path) if subset_path else None
        
        self.return_parent_pages = parent_document_retrieval
        self.llm_reranking = llm_reranking
        self.llm_reranking_sample_size = llm_reranking_sample_size
        self.top_n_retrieval = top_n_retrieval
        self.answering_model = answering_model
        self.parallel_requests = parallel_requests
        self.api_provider = api_provider
        self.openai_processor = APIProcessor(provider=api_provider)
        self.full_context = full_context

        self.answer_details = []
        self.detail_counter = 0
        self._lock = threading.Lock()

    def _load_questions(self, questions_file_path: Optional[Union[str, Path]]) -> List[Dict[str, str]]:
        if questions_file_path is None:
            return []
        with open(questions_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _format_retrieval_results(self, retrieval_results) -> str:
        """Format vector retrieval results into RAG context string"""
        if not retrieval_results:
            return ""
        
        context_parts = []
        for result in retrieval_results:
            document_name = result.get('document_name', 'Неизвестный документ')
            page_number = result['page']
            text = result['text']
            context_parts.append(f'Текст из документа "{document_name}", страница {page_number}: \n"""\n{text}\n"""')
            
        return "\n\n---\n\n".join(context_parts)

    def _extract_references(self, pages_list: list, document_name: str) -> list:
        # Загрузка данных о документах
        if self.subset_path is None:
            raise ValueError("subset_path необходим для обработки ссылок.")
        self.documents_df = pd.read_csv(self.subset_path)

        # Найти SHA1 документа из CSV
        matching_rows = self.documents_df[self.documents_df['document_name'] == document_name]
        if matching_rows.empty:
            document_sha1 = ""
        else:
            document_sha1 = matching_rows.iloc[0]['sha1']

        refs = []
        for page in pages_list:
            refs.append({"pdf_sha1": document_sha1, "page_index": page})
        return refs

    def _validate_page_references(self, claimed_pages: list, retrieval_results: list, min_pages: int = 2, max_pages: int = 8) -> list:
        """
        Проверяет, что все номера страниц, упомянутые в ответе LLM, действительно есть в результатах поиска.
        Если валидных ссылок меньше min_pages, добавляет верхние страницы из результатов поиска.
        """
        if claimed_pages is None:
            claimed_pages = []
        
        retrieved_pages = [result['page'] for result in retrieval_results]
        
        validated_pages = [page for page in claimed_pages if page in retrieved_pages]
        
        if len(validated_pages) < len(claimed_pages):
            removed_pages = set(claimed_pages) - set(validated_pages)
            print(f"Предупреждение: Удалено {len(removed_pages)} несуществующих ссылок на страницы: {removed_pages}")
        
        if len(validated_pages) < min_pages and retrieval_results:
            existing_pages = set(validated_pages)
            
            for result in retrieval_results:
                page = result['page']
                if page not in existing_pages:
                    validated_pages.append(page)
                    existing_pages.add(page)
                    
                    if len(validated_pages) >= min_pages:
                        break
        
        if len(validated_pages) > max_pages:
            print(f"Сокращение ссылок с {len(validated_pages)} до {max_pages} страниц")
            validated_pages = validated_pages[:max_pages]
        
        return validated_pages

    def get_answer_for_document(self, document_name: str, question: str, schema: str) -> dict:
        if self.llm_reranking:
            retriever = HybridRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )
        else:
            retriever = VectorRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )

        if self.full_context:
            retrieval_results = retriever.retrieve_all(document_name)
        else:           
            retrieval_results = retriever.retrieve_by_document_name(
                document_name=document_name,
                query=question,
                llm_reranking_sample_size=self.llm_reranking_sample_size,
                top_n=self.top_n_retrieval,
                return_parent_pages=self.return_parent_pages
            )
        
        if not retrieval_results:
            raise ValueError("Не найден релевантный контекст")
        
        rag_context = self._format_retrieval_results(retrieval_results)
        answer_dict = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema=schema,
            model=self.answering_model
        )
        self.response_data = self.openai_processor.response_data
        
        pages = answer_dict.get("relevant_pages", [])
        validated_pages = self._validate_page_references(pages, retrieval_results)
        answer_dict["relevant_pages"] = validated_pages
        answer_dict["references"] = self._extract_references(validated_pages, document_name)
        return answer_dict

    def _extract_documents_from_subset(self, question_text: str) -> list[str]:
        """Извлекает названия документов из вопроса, сопоставляя с документами в файле подмножества."""
        if not hasattr(self, 'documents_df'):
            if self.subset_path is None:
                raise ValueError("subset_path должен быть указан для использования извлечения из подмножества")
            self.documents_df = pd.read_csv(self.subset_path)
        
        found_documents = []
        document_names = sorted(self.documents_df['document_name'].unique(), key=len, reverse=True)
        
        for document in document_names:
            escaped_document = re.escape(document)
            
            pattern = rf'{escaped_document}(?:\W|$)'
            
            if re.search(pattern, question_text, re.IGNORECASE):
                found_documents.append(document)
                question_text = re.sub(pattern, '', question_text, flags=re.IGNORECASE)
        
        return found_documents

    def process_question(self, question: str, schema: str):
        # Пытаемся извлечь названия документов из вопроса
        extracted_documents = self._extract_documents_from_subset(question)
        
        # Если документы не найдены в вопросе, ищем в кавычках
        if len(extracted_documents) == 0:
            extracted_documents = re.findall(r'"([^"]*)"', question)
        
        if len(extracted_documents) == 0:
            # Если документы не указаны, выполняем поиск по всем документам
            return self.process_general_question(question, schema)
        
        if len(extracted_documents) == 1:
            document_name = extracted_documents[0]
            answer_dict = self.get_answer_for_document(document_name=document_name, question=question, schema=schema)
            return answer_dict
        else:
            return self.process_comparative_question(question, extracted_documents, schema)
    
    def process_general_question(self, question: str, schema: str) -> dict:
        """Обрабатывает вопрос без указания конкретного документа, ищет по всей базе."""
        if self.llm_reranking:
            retriever = HybridRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )
        else:
            retriever = VectorRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )
        
        # Поиск по всем документам
        retrieval_results = retriever.retrieve_by_query(
            query=question,
            llm_reranking_sample_size=self.llm_reranking_sample_size,
            top_n=self.top_n_retrieval,
            return_parent_pages=self.return_parent_pages
        )
        
        if not retrieval_results:
            raise ValueError("Не найден релевантный контекст")
        
        rag_context = self._format_retrieval_results(retrieval_results)
        answer_dict = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema=schema,
            model=self.answering_model
        )
        self.response_data = self.openai_processor.response_data
        
        # Группируем ссылки по документам
        document_pages = {}
        for result in retrieval_results:
            doc_name = result.get('document_name', 'unknown')
            if doc_name not in document_pages:
                document_pages[doc_name] = []
            document_pages[doc_name].append(result['page'])
        
        # Собираем все ссылки
        all_references = []
        for doc_name, pages in document_pages.items():
            refs = self._extract_references(pages, doc_name)
            all_references.extend(refs)
        
        answer_dict["references"] = all_references
        return answer_dict

    def _create_answer_detail_ref(self, answer_dict: dict, question_index: int) -> str:
        """Создает ID ссылки для деталей ответа и сохраняет детали"""
        ref_id = f"#/answer_details/{question_index}"
        with self._lock:
            self.answer_details[question_index] = {
                "step_by_step_analysis": answer_dict['step_by_step_analysis'],
                "reasoning_summary": answer_dict['reasoning_summary'],
                "relevant_pages": answer_dict['relevant_pages'],
                "response_data": self.response_data,
                "self": ref_id
            }
        return ref_id

    def _calculate_statistics(self, processed_questions: List[dict], print_stats: bool = False) -> dict:
        """Рассчитывает статистику по обработанным вопросам."""
        total_questions = len(processed_questions)
        error_count = sum(1 for q in processed_questions if "error" in q)
        na_count = sum(1 for q in processed_questions if (q.get("value") if "value" in q else q.get("answer")) == "N/A")
        success_count = total_questions - error_count - na_count
        if print_stats:
            print(f"\nИтоговая статистика обработки:")
            print(f"Всего вопросов: {total_questions}")
            print(f"Ошибок: {error_count} ({(error_count/total_questions)*100:.1f}%)")
            print(f"Ответов N/A: {na_count} ({(na_count/total_questions)*100:.1f}%)")
            print(f"Успешно отвечено: {success_count} ({(success_count/total_questions)*100:.1f}%)\n")
        
        return {
            "total_questions": total_questions,
            "error_count": error_count,
            "na_count": na_count,
            "success_count": success_count
        }

    def process_questions_list(self, questions_list: List[dict], output_path: str = None, submission_file: bool = False, team_email: str = "", submission_name: str = "", pipeline_details: str = "") -> dict:
        total_questions = len(questions_list)
        # Add index to each question so we know where to write the answer details
        questions_with_index = [{**q, "_question_index": i} for i, q in enumerate(questions_list)]
        self.answer_details = [None] * total_questions  # Preallocate list for answer details
        processed_questions = []
        parallel_threads = self.parallel_requests

        if parallel_threads <= 1:
            for question_data in tqdm(questions_with_index, desc="Processing questions"):
                processed_question = self._process_single_question(question_data)
                processed_questions.append(processed_question)
                if output_path:
                    self._save_progress(processed_questions, output_path, submission_file=submission_file, team_email=team_email, submission_name=submission_name, pipeline_details=pipeline_details)
        else:
            with tqdm(total=total_questions, desc="Processing questions") as pbar:
                for i in range(0, total_questions, parallel_threads):
                    batch = questions_with_index[i : i + parallel_threads]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                        # executor.map will return results in the same order as the input list.
                        batch_results = list(executor.map(self._process_single_question, batch))
                    processed_questions.extend(batch_results)
                    
                    if output_path:
                        self._save_progress(processed_questions, output_path, submission_file=submission_file, team_email=team_email, submission_name=submission_name, pipeline_details=pipeline_details)
                    pbar.update(len(batch_results))
        
        statistics = self._calculate_statistics(processed_questions, print_stats = True)
        
        return {
            "questions": processed_questions,
            "answer_details": self.answer_details,
            "statistics": statistics
        }

    def _process_single_question(self, question_data: dict) -> dict:
        question_index = question_data.get("_question_index", 0)
        
        question_text = question_data.get("question")
        schema = question_data.get("schema")
        try:
            answer_dict = self.process_question(question_text, schema)
            
            if "error" in answer_dict:
                detail_ref = self._create_answer_detail_ref({
                    "step_by_step_analysis": None,
                    "reasoning_summary": None,
                    "relevant_pages": None
                }, question_index)
                return {
                    "question_text": question_text,
                    "kind": schema,
                    "value": None,
                    "references": [],
                    "error": answer_dict["error"],
                    "answer_details": {"$ref": detail_ref}
                }
            detail_ref = self._create_answer_detail_ref(answer_dict, question_index)
            return {
                "question_text": question_text,
                "kind": schema,
                "value": answer_dict.get("final_answer"),
                "references": answer_dict.get("references", []),
                "answer_details": {"$ref": detail_ref},
            }
        except Exception as err:
            return self._handle_processing_error(question_text, schema, err, question_index)

    def _handle_processing_error(self, question_text: str, schema: str, err: Exception, question_index: int) -> dict:
        """
        Handle errors during question processing.
        Log error details and return a dictionary containing error information.
        """
        import traceback
        error_message = str(err)
        tb = traceback.format_exc()
        error_ref = f"#/answer_details/{question_index}"
        error_detail = {
            "error_traceback": tb,
            "self": error_ref
        }
        
        with self._lock:
            self.answer_details[question_index] = error_detail
        
        print(f"Error encountered processing question: {question_text}")
        print(f"Error type: {type(err).__name__}")
        print(f"Error message: {error_message}")
        print(f"Full traceback:\n{tb}\n")
        
        return {
            "question_text": question_text,
            "kind": schema,
            "value": None,
            "references": [],
            "error": f"{type(err).__name__}: {error_message}",
            "answer_details": {"$ref": error_ref},
        }

    def _post_process_submission_answers(self, processed_questions: List[dict]) -> List[dict]:
        """
        Post-process answers for submission format:
        1. Convert page indices from one-based to zero-based
        2. Clear references for N/A answers
        3. Format answers according to submission schema
        4. Include step_by_step_analysis from answer details
        """
        submission_answers = []
        
        for q in processed_questions:
            question_text = q.get("question_text") or q.get("question")
            kind = q.get("kind") or q.get("schema")
            value = "N/A" if "error" in q else (q.get("value") if "value" in q else q.get("answer"))
            references = q.get("references", [])
            
            answer_details_ref = q.get("answer_details", {}).get("$ref", "")
            step_by_step_analysis = None
            if answer_details_ref and answer_details_ref.startswith("#/answer_details/"):
                try:
                    index = int(answer_details_ref.split("/")[-1])
                    if 0 <= index < len(self.answer_details) and self.answer_details[index]:
                        step_by_step_analysis = self.answer_details[index].get("step_by_step_analysis")
                except (ValueError, IndexError):
                    pass
            
            # Clear references if value is N/A
            if value == "N/A":
                references = []
            else:
                # Convert page indices from one-based to zero-based (competition requires 0-based page indices, but for debugging it is easier to use 1-based)
                references = [
                    {
                        "pdf_sha1": ref["pdf_sha1"],
                        "page_index": ref["page_index"] - 1
                    }
                    for ref in references
                ]
            
            submission_answer = {
                "question_text": question_text,
                "kind": kind,
                "value": value,
                "references": references,
            }
            
            if step_by_step_analysis:
                submission_answer["reasoning_process"] = step_by_step_analysis
            
            submission_answers.append(submission_answer)
        
        return submission_answers

    def _save_progress(self, processed_questions: List[dict], output_path: Optional[str], submission_file: bool = False, team_email: str = "", submission_name: str = "", pipeline_details: str = ""):
        if output_path:
            statistics = self._calculate_statistics(processed_questions)
            
            # Prepare debug content
            result = {
                "questions": processed_questions,
                "answer_details": self.answer_details,
                "statistics": statistics
            }
            output_file = Path(output_path)
            debug_file = output_file.with_name(output_file.stem + "_debug" + output_file.suffix)
            with open(debug_file, 'w', encoding='utf-8') as file:
                json.dump(result, file, ensure_ascii=False, indent=2)
            
            if submission_file:
                # Post-process answers for submission
                submission_answers = self._post_process_submission_answers(processed_questions)
                submission = {
                    "answers": submission_answers,
                    "team_email": team_email,
                    "submission_name": submission_name,
                    "details": pipeline_details
                }
                with open(output_file, 'w', encoding='utf-8') as file:
                    json.dump(submission, file, ensure_ascii=False, indent=2)

    def process_all_questions(self, output_path: str = 'questions_with_answers.json', team_email: str = "79250515615@yandex.com", submission_name: str = "Ilia_Ris SO CoT + Parent Document Retrieval", submission_file: bool = False, pipeline_details: str = ""):
        result = self.process_questions_list(
            self.questions,
            output_path,
            submission_file=submission_file,
            team_email=team_email,
            submission_name=submission_name,
            pipeline_details=pipeline_details
        )
        return result

    def process_comparative_question(self, question: str, documents: List[str], schema: str) -> dict:
        """
        Обрабатывает вопрос, касающийся нескольких документов параллельно:
        1. Перефразирует сравнительный вопрос в отдельные вопросы
        2. Обрабатывает каждый отдельный вопрос, используя параллельные потоки
        3. Объединяет результаты в итоговый сравнительный ответ
        """
        # Шаг 1: Перефразируем сравнительный вопрос
        rephrased_questions = self.openai_processor.get_rephrased_questions(
            original_question=question,
            documents=documents  # Заменили companies на documents
        )
        
        individual_answers = {}
        aggregated_references = []
        
        # Шаг 2: Обрабатываем каждый отдельный вопрос параллельно
        def process_document_question(document: str) -> tuple[str, dict]:
            """Вспомогательная функция для обработки вопроса по одному документу"""
            sub_question = rephrased_questions.get(document)
            if not sub_question:
                raise ValueError(f"Не удалось сгенерировать подвопрос для документа: {document}")
            
            answer_dict = self.get_answer_for_document(
                document_name=document, 
                question=sub_question, 
                schema="text"  # Изменили schema с "number" на "text"
            )
            return document, answer_dict

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_document = {
                executor.submit(process_document_question, document): document 
                for document in documents
            }
            
            for future in concurrent.futures.as_completed(future_to_document):
                try:
                    document, answer_dict = future.result()
                    individual_answers[document] = answer_dict
                    
                    document_references = answer_dict.get("references", [])
                    aggregated_references.extend(document_references)
                except Exception as e:
                    document = future_to_document[future]
                    print(f"Ошибка обработки документа {document}: {str(e)}")
                    raise
        
        # Удаляем дублирующиеся ссылки
        unique_refs = {}
        for ref in aggregated_references:
            key = (ref.get("pdf_sha1"), ref.get("page_index"))
            unique_refs[key] = ref
        aggregated_references = list(unique_refs.values())
        
        # Шаг 3: Получаем сравнительный ответ, используя все индивидуальные ответы
        comparative_answer = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=individual_answers,
            schema="comparative",
            model=self.answering_model
        )
        self.response_data = self.openai_processor.response_data
        
        comparative_answer["references"] = aggregated_references
        return comparative_answer
    