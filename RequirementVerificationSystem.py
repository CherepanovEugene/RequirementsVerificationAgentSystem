import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_gigachat import GigaChat, GigaChatEmbeddings  # Для работы с GigaChat
from langchain.prompts import PromptTemplate

# Создание папки для логов
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)

# Функция для настройки логирования
def setup_logger(agent_name):
    logger = logging.getLogger(agent_name)
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join(log_folder, f"{agent_name}.log")
    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

# --- Загрузка API-ключа ---
load_dotenv()  # Загружаем переменные окружения из .env файла
# --- Загрузка API-ключа ---
load_dotenv()  # Загружаем переменные окружения из .env файла
gigachat_api_key = os.getenv("GIGACHAT_API_KEY")  # Получаем API-ключ GigaChat

# --- Инициализация GigaChat ---
giga = GigaChat(
    credentials=gigachat_api_key,  # Передаём API-ключ для авторизации
    scope="GIGACHAT_API_PERS",     # Указываем область применения API (например, персональный доступ)
    model="GigaChat-Max",              # Задаём модель GigaChat для использования
    streaming=False,               # Отключаем потоковую передачу данных (ответ будет получен целиком)
    verify_ssl_certs=False         # Игнорируем проверку SSL-сертификатов (не рекомендуется в продакшене)
)

# Логгеры для каждого агента
communicator_logger = setup_logger("communicator_agent")
verifier_logger = setup_logger("verifier_agent")
expert_logger = setup_logger("expert_agent")
final_verifier_logger = setup_logger("final_verifier_agent")
orchestrator_logger = setup_logger("orchestrator_agent")

# --- Агент-Коммуникатор ---
def communicator_agent(requirement):
    communicator_logger.info("[Communicator] Получено требование: %s", requirement)
    return requirement

# --- Агент-Верификатор ---
def verifier_agent(requirement):
    prompt = PromptTemplate(
        input_variables=["requirement"],
        template="Проверьте следующее требование на полноту, ясность и непротиворечивость: {requirement}."
    )
    chain = prompt | giga
    result = chain.invoke({"requirement": requirement})
    verifier_logger.info("[Verifier] Результат проверки: %s", result.content)
    return result.content

# --- Консилиум AI-Агентов ---
def expert_agent(role, requirement):
    prompt = PromptTemplate(
        input_variables=["role", "requirement"],
        template="Как эксперт по {role}, оцените требование: {requirement}. Если оно не относится к вашей области, ответьте 'не относится'."
    )
    chain = prompt | giga
    result = chain.invoke({"role": role, "requirement": requirement})
    response_text = result.content if hasattr(result, 'content') else str(result)
    expert_logger.info("[%s] Мнение: %s", role, response_text)
    return response_text if "не относится" not in response_text.lower() else None

# --- Финальная проверка Агентом-Верификатором ---
def final_verification(requirement, expert_feedbacks):
    feedback_str = " \n".join([fb for fb in expert_feedbacks if fb])
    prompt = PromptTemplate(
        input_variables=["requirement", "feedback"],
        template="Учитывая исходное требование: {requirement} и мнения экспертов: {feedback}, сформируйте окончательную формулировку."
    )
    chain = prompt | giga
    result = chain.invoke({"requirement": requirement, "feedback": feedback_str})
    final_verifier_logger.info("[Final Verifier] Итоговое требование: %s", result.content)
    return result.content

# --- Агент-Оркестратор ---
def orchestrator(final_requirement):
    orchestrator_logger.info("[Orchestrator] Итоговое требование передано для реализации: %s", final_requirement)
    return final_requirement

# --- Основной процесс ---
def main():
    requirement = "Система должна поддерживать резервное копирование данных."

    # 1. Коммуникатор получает требование
    received_requirement = communicator_agent(requirement)

    # 2. Верификатор анализирует требование
    verification_result = verifier_agent(received_requirement)

    # 3. Консилиум AI-Агентов
    experts = [
        "кибербезопасность",
        "ИТ-инфраструктура и DevOps",
        "ИТ-Архитектура",
        "разработка и детальная архитектура приложений"
    ]
    expert_feedbacks = [expert_agent(role, received_requirement) for role in experts]

    # 4. Финальная проверка
    final_requirement = final_verification(received_requirement, expert_feedbacks)

    # 5. Оркестратор завершает процесс
    orchestrator(final_requirement)

if __name__ == "__main__":
    main()