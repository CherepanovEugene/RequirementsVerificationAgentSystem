# Импорт библиотек
import os                                  # Работа с файловой системой
import logging                             # Логирование действий
from dotenv import load_dotenv             # Загрузка переменных окружения из файла .env
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Разделение текста
from langchain_gigachat import GigaChatEmbeddings, GigaChat         # Эмбеддинги и LLM GigaChat
from langchain_community.vectorstores import FAISS                  # Векторное хранилище
from langgraph.graph import StateGraph, END                         # Графы для агентов
from langchain.schema import Document                               # Работа с документами
from pydantic import BaseModel, Field                               # Описание моделей данных
from typing import List, Dict, Optional                             # Типизация данных
import fitz                                                         # Работа с PDF (PyMuPDF)
from fpdf import FPDF, XPos, YPos                                   # Генерация PDF-отчётов

# Загрузка переменных из .env
load_dotenv()

# Настройка удобного логирования
# Настройка общего логирования
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)

def setup_logger(agent_name):
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
llm = GigaChat(credentials=os.getenv("GIGACHAT_API_KEY"), model="GigaChat-Max", verify_ssl_certs=False)
embeddings = GigaChatEmbeddings(credentials=os.getenv("GIGACHAT_API_KEY"), model="Embeddings", verify_ssl_certs=False)

# Модель состояния графа
class GraphState(BaseModel):
    file_path: str                                       # Путь к PDF-документу
    questions: List[str]                                 # Список вопросов для анализа
    docs: List[Document] = Field(default_factory=list)   # Части документа
    vectorstore: Optional[FAISS] = None                  # Векторное хранилище
    analysis_result: Dict[str, List[str]] = Field(default_factory=dict)  # Результаты анализа

    class Config:
        arbitrary_types_allowed = True                   # Разрешение произвольных типов

# Агент-Парсер: разбивает PDF и сохраняет части в виде эмбеддингов
def parser_agent(state: GraphState):
    parser_logger.info("Начало работы агента парсинга")

    doc = fitz.open(state.file_path)
    full_text = ""

    # Извлекаем текст из всех страниц документа
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        full_text += page_text
        parser_logger.info(f"Обработана страница {page_num + 1}, размер текста: {len(page_text)} символов")

    # Первичное деление по абзацам
    raw_paragraphs = [para.strip() for para in full_text.split('\n\n') if para.strip()]
    parser_logger.info(f"Документ разделён на {len(raw_paragraphs)} абзацев (первичное деление)")

    # Дополнительное разбиение длинных абзацев
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    docs = []
    for para in raw_paragraphs:
        # Если абзац слишком длинный, разбиваем дополнительно
        if len(para.split()) > 400:
            sub_docs = splitter.create_documents([para])
            docs.extend(sub_docs)
            parser_logger.info(f"Абзац был дополнительно разделён на {len(sub_docs)} частей из-за большой длины.")
        else:
            docs.append(Document(page_content=para))

    parser_logger.info(f"Итоговое количество фрагментов: {len(docs)}")

    # Генерируем эмбеддинги и сохраняем в FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)

    state.docs = docs
    state.vectorstore = vectorstore

    parser_logger.info("Завершение работы агента парсинга")
    return state

# Агент-Анализатор: проверяет каждую часть документа через LLM и ищет релевантные вопросу фрагменты
# def analysis_agent(state: GraphState):
#     analysis_logger.info("Начало работы агента анализа")
#     state.analysis_result = {question: [] for question in state.questions}
#
#     for idx, doc in enumerate(state.docs):
#         analysis_logger.info(f"Анализируется часть документа №{idx + 1}")
#         for question in state.questions:
#             prompt = (f"В данной части документа может быть ответ на вопрос: \"{question}\"? "
#                       f"Ответь одним коротким предложением. Часть документа: {doc.page_content[:200]}...")
#             response = llm.invoke(prompt).content
#             analysis_logger.info(f"Ответ LLM: {response}")
#             if "да" in response.lower():
#                 state.analysis_result[question].append(doc.page_content)
#                 analysis_logger.info(f"Фрагмент добавлен для вопроса: {question}")
#
#     analysis_logger.info("Завершение работы агента анализа")
#     return state


# Агент-Анализатор: проверяет каждую часть документа через LLM и ищет ответы на вопросы в каждом фрагменте
def analysis_agent(state: GraphState):
    analysis_logger.info("Начало работы агента анализа")
    state.analysis_result = {question: [] for question in state.questions}

    for idx, doc in enumerate(state.docs):
        analysis_logger.info(f"Анализируется фрагмент №{idx + 1}")
        for question in state.questions:
            prompt = (
                f"Используя следующий фрагмент документа, ответь кратко и точно на вопрос.\n\n"
                f"Вопрос: {question}\n\n"
                f"Фрагмент документа:\n{doc.page_content}\n\n"
                f"Если информации недостаточно, ответь: 'Ответ не найден'."
            )

            response = llm.invoke(prompt).content.strip()
            analysis_logger.info(f"Вопрос: {question} | Ответ LLM: {response}")

            if response.lower() != "ответ не найден":
                state.analysis_result[question].append({
                    "answer": response,
                    "fragment": doc.page_content
                })

    analysis_logger.info("Завершение работы агента анализа")
    return state


# Агент-Отчёт: формирует отчёт в PDF: вопросы и фрагменты
# def report_agent(state: GraphState):
#     report_logger.info("Начало работы агента отчёта")
#     pdf = FPDF()
#     pdf.add_page()
#
#     # Подключение обычного и жирного шрифтов
#     pdf.add_font("DejaVu", "", "fonts/DejaVuSans.ttf")
#     pdf.add_font("DejaVu", "B", "fonts/DejaVuSans-Bold.ttf")
#
#     # Титульный заголовок отчёта
#     pdf.set_font("DejaVu", size=16)
#     pdf.set_text_color(0, 102, 204)  # Синий цвет
#     pdf.cell(0, 12, "Отчёт по анализу документа", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
#     pdf.ln(8)
#
#     # Перебор вопросов и фрагментов
#     for question, fragments in state.analysis_result.items():
#         # Запись вопроса жирным шрифтом
#         pdf.set_font("DejaVu", style='B', size=13)
#         pdf.set_text_color(0, 0, 0)
#         pdf.multi_cell(0, 10, text=f"Вопрос: {question}", align='L')
#         pdf.ln(2)
#
#         # Запись фрагментов ответа
#         if fragments:
#             for idx, fragment in enumerate(fragments, start=1):
#                 cleaned_fragment = fragment.replace('\n', ' ').strip()
#                 pdf.set_font("DejaVu", size=12)
#                 pdf.set_text_color(40, 40, 40)
#                 pdf.multi_cell(0, 8, text=f"{idx}. {cleaned_fragment}", align='L')
#                 pdf.ln(3)
#         else:
#             pdf.set_font("DejaVu", style='I', size=12)
#             pdf.set_text_color(255, 0, 0)
#             pdf.cell(0, 8, text="Ответ не найден.", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
#             pdf.ln(3)
#
#         pdf.ln(5)
#
#     # Колонтитул с номером страницы
#     pdf.set_y(-15)
#     pdf.set_font("DejaVu", size=10)
#     pdf.set_text_color(100, 100, 100)
#     pdf.cell(0, 10, f'Страница {pdf.page_no()}', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
#
#     # Сохранение отчёта
#     output_path = "analysis_report.pdf"
#     pdf.output(output_path)
#
#     report_logger.info(f"Агент отчёта завершил работу. Отчёт сохранён в {output_path}")
#     print(f"Отчёт сохранён в {output_path}")
#     return state

# Агент-Отчёт: формирует отчёт в PDF: вопросы, ответы и фрагменты
def report_agent(state: GraphState):
    report_logger.info("Начало работы агента отчёта")
    pdf = FPDF()
    pdf.add_page()

    # Подключение обычного и жирного шрифтов
    pdf.add_font("DejaVu", "", "fonts/DejaVuSans.ttf")
    pdf.add_font("DejaVu", "B", "fonts/DejaVuSans-Bold.ttf")

    # Основной заголовок отчёта
    pdf.set_font("DejaVu", size=16)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 12, "Отчёт по анализу документа", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(8)

    # Перебор вопросов и ответов
    for question, items in state.analysis_result.items():
        pdf.set_font("DejaVu", style='B', size=13)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 10, text=f"Вопрос: {question}", align='L')
        pdf.ln(2)

        if items:
            for idx, item in enumerate(items, start=1):
                answer = item["answer"]
                fragment = item["fragment"].replace('\n', ' ').strip()

                # Ответ выделяем зелёным
                pdf.set_font("DejaVu", style='B', size=12)
                pdf.set_text_color(0, 128, 0)
                pdf.multi_cell(0, 8, text=f"Ответ: {answer}", align='L')
                pdf.ln(1)

                # Исходный фрагмент документа серым цветом
                pdf.set_font("DejaVu", size=11)
                pdf.set_text_color(40, 40, 40)
                pdf.multi_cell(0, 8, text=f"Фрагмент: {fragment}", align='L')
                pdf.ln(4)
        else:
            pdf.set_font("DejaVu", style='I', size=12)
            pdf.set_text_color(255, 0, 0)
            pdf.cell(0, 8, text="Ответ не найден.", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
            pdf.ln(3)

        pdf.ln(5)

    # Колонтитул с номером страницы
    pdf.set_y(-15)
    pdf.set_font("DejaVu", size=10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f'Страница {pdf.page_no()}', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

    # Сохранение отчёта
    output_path = "analysis_report.pdf"
    pdf.output(output_path)

    report_logger.info(f"Агент отчёта завершил работу. Отчёт сохранён в {output_path}")
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
        "Из каких этапов состоит процесс управления уязвимостями?",
        "Как проводится аудит безопасности?",
        "Каков срок действия документа?"
    ]

    communicate(file_name, questions)