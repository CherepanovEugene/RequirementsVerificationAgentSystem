# Requirements Verification Agent System

Мультиагентная система для поиска ответов на несколько вопросов одновременно по нескольким документам (PDF/DOCX/TXT) с устойчивостью к лимитам GigaChat и удобным PDF-отчётом.

## Возможности

✅ Принимает папку или список файлов (PDF/DOCX/TXT).

✅ Разбивает документы на фрагменты с метаданными: file, page, id.

✅ Строит единое FAISS-хранилище (GigaChat Embeddings).

✅ Для каждого вопроса: быстрый векторный поиск релевантных фрагментов (MMR/Similarity) → один запрос к LLM.

✅ Возвращает строгий JSON для каждого вопроса:
```
{
  "answer": "…или not_found",
  "confidence": 0.0,
  "citations": [{"id": 123, "file": "a.pdf", "page": 4}],
  "quotes": ["...цитата..."]
}
```
   
✅ Генерирует PDF-отчёт с только найденными ответами, цитатами и источниками.

✅ Устойчивость к API-лимитам.


## Архитектура
### Граф узлов 
```
[ parser ]  ->  [ analysis ]  ->  [ report ]  ->  END
```
* parser — парсинг (PDF/DOCX/TXT), разбиение, построение FAISS.

* analysis — пред-эмбеддинг вопросов → поиск по вектору → 1 LLM-вызов на вопрос.

* report — PDF-отчёт с ответами, цитатами, источниками.
### Состояние
```
class GraphState(BaseModel):
    file_paths: List[str]                    # входные документы
    questions: List[str]                     # список вопросов
    docs: List[Document]                     # фрагменты с метаданными (file, page, id)
    vectorstore: Optional[FAISS]             # векторное хранилище
    analysis_result: Dict[str, List[Dict]]   # ответы по вопросам (answer, confidence, citations, quotes)
```
## Поток выполнения
1. Парсинг
* PDF: PyMuPDF (fitz).
* DOCX: docx2txt → fallback python-docx → fallback zipfile (чтение word/document.xml).
* TXT: чтение файла.
* Деление на абзацы → дополнительный RecursiveCharacterTextSplitter.
* Проставление метаданных: id, file, page.
* Перед эмбеддингом — усечение фрагментов по «словам» до безопасной длины (аппроксимация токенов).
2. FAISS
* FAISS.from_texts(texts, embedding, metadatas).
3. Анализ
* Пред-эмбеддинг вопросов (последовательно, с backoff) → избегаем 429 при поиске.
* Поиск контекста: similarity_search_by_vector() (локально по FAISS, без API).
* Компактный контекст-пакет (ограничение по символам).
* Один LLM-запрос на вопрос с требованием строгого JSON.
* Если LLM не вернул источники — подставляем ближайший релевантный.
4. Отчёт
* Только найденные ответы.
* Ответы с confidence, цитаты (до 3), источники (file, page, id).
* Шрифты: DejaVuSans.ttf, DejaVuSans-Bold.ttf.
* Предварительная нормализация строк (перенос длинных «слов») для fpdf2.
## Требования
* Python 3.10+ (проверено на 3.13).
* Библиотеки:
```
pip install \
  langchain langchain-community langgraph python-dotenv \
  langchain-gigachat gigachat \
  faiss-cpu \
  PyMuPDF \
  fpdf2 \
  docx2txt python-docx
```
* .env:
```
GIGACHAT_API_KEY=ваш_ключ
```
* Шрифты (папка fonts/ рядом со скриптом):
```
fonts/DejaVuSans.ttf
fonts/DejaVuSans-Bold.ttf
```
## Установка
1. Установите зависимости (см. выше).
2. Скопируйте шрифты в fonts/.
3. Создайте .env и впишите GIGACHAT_API_KEY.
## Структура проекта
```
project_root/
├─ docs/                       # входные файлы (pdf/docx/txt)
│   ├─ a.pdf
│   └─ b.docx
├─ fonts/
│   ├─ DejaVuSans.ttf
│   └─ DejaVuSans-Bold.ttf
├─ logs/                       # генерируется автоматически
├─ .env
├─ DocAnalysisAgents.py        # основной скрипт
└─ analysis_report.pdf         # результат работы
```
