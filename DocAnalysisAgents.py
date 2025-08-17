# Импорт библиотек
import os                                  # Работа с файловой системой
import logging                             # Логирование действий
import json                                # Разбор JSON-ответов
from dotenv import load_dotenv             # Загрузка переменных окружения из файла .env
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Разделение текста
from langchain_gigachat import GigaChat, GigaChatEmbeddings         # Эмбеддинги и LLM GigaChat
from langchain_community.vectorstores import FAISS                  # Векторное хранилище
from langgraph.graph import StateGraph, END                         # Графы для агентов
from langchain.schema import Document                               # Работа с документами
from pydantic import BaseModel, Field                               # Описание моделей данных
from typing import List, Dict, Optional                             # Типизация данных
import fitz                                                         # Работа с PDF (PyMuPDF)
from fpdf import FPDF, XPos, YPos                                   # Генерация PDF-отчётов

# Загрузка переменных из .env
load_dotenv()

def setup_logger(agent_name):
    log_folder = "logs"
    os.makedirs(log_folder, exist_ok=True)
    logger = logging.getLogger(agent_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(log_folder, f'{agent_name}.log'))
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(formatter)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(file_handler)
    return logger

# Инициализация логгеров для каждого агента
parser_logger = setup_logger('parser_agent')
analysis_logger = setup_logger('analysis_agent')
report_logger = setup_logger('report_agent')
communication_logger = setup_logger('communication_agent')

# Инициализация модели GigaChat и эмбеддингов
llm = GigaChat(credentials=os.getenv("GIGACHAT_API_KEY"), model="GigaChat-Max", verify_ssl_certs=False, timeout=60)
embeddings = GigaChatEmbeddings(credentials=os.getenv("GIGACHAT_API_KEY"), model="Embeddings", verify_ssl_certs=False)

# Модель состояния графа
class GraphState(BaseModel):
    file_path: str                                       # Путь к PDF-документу
    questions: List[str]                                 # Список вопросов для анализа
    docs: List[Document] = Field(default_factory=list)   # Части документа
    vectorstore: Optional[FAISS] = None                  # Векторное хранилище
    analysis_result: Dict[str, List[Dict[str, str]]] = Field(default_factory=dict)  # Результаты анализа

    class Config:
        arbitrary_types_allowed = True                   # Разрешение произвольных типов

# Агент-Парсер: разбивает PDF и сохраняет части в виде эмбеддингов
def parser_agent(state: GraphState):
    parser_logger.info("Начало работы агента парсинга")
    doc = fitz.open(state.file_path)
    full_text = ""
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        full_text += page_text
        parser_logger.info(f"Обработана страница {page_num + 1}, размер текста: {len(page_text)} символов")
    raw_paragraphs = [para.strip() for para in full_text.split('\n\n') if para.strip()]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = []
    for para in raw_paragraphs:
        if len(para.split()) > 400:
            sub_docs = splitter.create_documents([para])
            docs.extend(sub_docs)
            parser_logger.info(f"Абзац разбит на {len(sub_docs)} частей.")
        else:
            docs.append(Document(page_content=para))
    parser_logger.info(f"Итоговое количество фрагментов: {len(docs)}")
    vectorstore = FAISS.from_documents(docs, embeddings)
    state.docs = docs
    state.vectorstore = vectorstore
    parser_logger.info("Завершение работы агента парсинга")
    return state

# Агент-Анализатор: ищет ответы на все вопросы одним запросом
def analysis_agent(state: GraphState):
    analysis_logger.info("Начало работы агента анализа")
    # Инициализация результата
    state.analysis_result = {q: [] for q in state.questions}
    for idx, doc in enumerate(state.docs):
        analysis_logger.info(f"Анализируется фрагмент №{idx + 1}")
        # Составляем единый запрос с несколькими вопросами
        questions_block = '\n'.join([f"{i+1}. {q}" for i, q in enumerate(state.questions)])
        prompt = (
            f"Используя следующий фрагмент документа, ответь кратко и точно на вопросы ниже в формате JSON."
            f"\n\nВопросы:\n{questions_block}\n\n"
            f"Фрагмент документа:\n{doc.page_content}\n\n"
            f"Возврати JSON-объект, где ключи — номера вопросов ('1', '2', ...) или текст вопросов, а значения — ответы."
        )
        try:
            response_text = llm.invoke(prompt, config={"timeout": 60}).content.strip()
            analysis_logger.info(f"Ответ LLM: {response_text}")
            # Парсим JSON-ответ
            answers = json.loads(response_text)
            # Сохраняем найденные ответы и фрагмент
            for key, ans in answers.items():
                if ans and ans.lower() != 'ответ не найден':
                    # определяем исходный вопрос
                    question = state.questions[int(key)-1] if key.isdigit() else key
                    state.analysis_result[question].append({'answer': ans, 'fragment': doc.page_content})
        except Exception as e:
            analysis_logger.error(f"Ошибка анализа фрагмента: {e}")
    analysis_logger.info("Завершение работы агента анализа")
    return state

# Агент-Отчёт: выводит лишь найденные ответы
def report_agent(state: GraphState):
    report_logger.info("Начало работы агента отчёта")
    pdf = FPDF()

    # — НАТИВНО: сразу открываем страницу —
    pdf.add_page()

    # — Регистрируем шрифты (без deprecated параметра uni) —
    pdf.add_font("DejaVu", "", "fonts/DejaVuSans.ttf")
    pdf.add_font("DejaVu", "B", "fonts/DejaVuSans-Bold.ttf")
    # если нужен курсив, заранее добавьте DejaVuSans-Oblique.ttf
    # pdf.add_font("DejaVu", "I", "fonts/DejaVuSans-Oblique.ttf")

    # — Заголовок отчёта —
    pdf.set_font("DejaVu", "B", 16)
    pdf.set_text_color(0, 102, 204)
    # format: cell(width, height, txt, border=0, ln=1→перенос, align='C')
    pdf.cell(0, 12, "Отчёт по анализу документа", 0, 1, "C")
    pdf.ln(5)

    # — Основное тело отчёта —
    for question, items in state.analysis_result.items():
        pdf.set_font("DejaVu", "B", 13)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, f"Вопрос: {question}", 0, 1)
        if items:
            for item in items:
                pdf.set_font("DejaVu", "B", 12)
                pdf.set_text_color(0, 128, 0)
                pdf.multi_cell(0, 8, f"Ответ: {item['answer']}")
                pdf.ln(1)

                pdf.set_font("DejaVu", "", 11)
                pdf.set_text_color(40, 40, 40)
                frag = item['fragment'].replace('\n', ' ').strip()
                pdf.multi_cell(0, 8, f"Фрагмент: {frag}")
                pdf.ln(4)
        else:
            pdf.set_font("DejaVu", "", 12)
            pdf.set_text_color(255, 0, 0)
            pdf.cell(0, 8, "Ответ не найден.", 0, 1)
        pdf.ln(5)

    # — Колонтитул с номером страницы —
    pdf.set_y(-15)
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f"Страница {pdf.page_no()}", 0, 0, "C")

    # — Сохранение —
    output_path = "analysis_report.pdf"
    pdf.output(output_path)
    report_logger.info(f"Отчёт сохранён в {output_path}")
    print(f"Отчёт сохранён в {output_path}")

    return state

# Агент-Коммуникатор: управляет графом агентов
def communicate(file_name, questions):
    communication_logger.info("Запуск коммуникационного агента")
    graph = StateGraph(GraphState)
    graph.add_node("parser", parser_agent)
    graph.add_node("analysis", analysis_agent)
    graph.add_node("report", report_agent)
    graph.set_entry_point("parser")
    graph.add_edge("parser", "analysis")
    graph.add_edge("analysis", "report")
    graph.add_edge("report", END)

    app = graph.compile()
    file_path = os.path.join("docs", file_name)
    state = GraphState(file_path=file_path, questions=questions)
    app.invoke(state)
    communication_logger.info("Коммуникационный агент завершил работу")

# Запуск всей системы (пример)
if __name__ == "__main__":
    file_name = "Регламент_управления_уязвимостями.pdf"
    questions = [
        "Назначения сотрудников на роли процесса закреплены?",
        "Для процесса установлены показатели КРІ?",
        "В описании процесса явно описаны Роли и Ответственность участников?"
        "Формируются отчеты ли по процессу, отражающие достижение КРІ процесса?"
    ]
    communicate(file_name, questions)
