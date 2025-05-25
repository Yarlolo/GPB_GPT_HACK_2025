import asyncio
import logging
import os
from typing import TypedDict
from enum import Enum
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class Position(str, Enum):
    DATA_SCIENTIST = "Data Scientist"
    DATA_ENGINEER = "Data Engineer"
    DATA_ANALYST = "Data Analyst"
    MLOPS_ENGINEER = "MLOps Engineer"
    PROJECT_MANAGER = "Project Manager"
    UNSUITABLE = "Некомпетентный соискатель"


class InterviewState(str, Enum):
    WAITING_FIRST_MESSAGE = "waiting_first_message"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class UserData(TypedDict):
    state: InterviewState
    position: Position
    message_count: int
    chat_history: list[dict]


bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
users: dict[int, UserData] = {}

llm = ChatOpenAI(
    model="leon-se/gemma-3-27b-it-FP8-Dynamic",
    openai_api_base="https://51.250.28.28:10000/gpb_gpt_hack_2025/v1",
    openai_api_key="EMPTY",
    temperature=0.7,
    max_tokens=512, )


async def generate_first_question() -> str:
    """Генерация первого вопроса"""
    prompt = """
    Сгенерируй ОДИН профессиональный вопрос для кандидата, который:
    1. Попросит рассказать о профессиональных навыках и опыте
    2. Будет нейтральным и не наводящим
    3. Не будет содержать упоминаний о конкретных позициях

    Формат: только вежливый вопрос с кратким приветствием и благодарностью за отклик кандидата, без пояснений.
    """
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        logging.error(f"Ошибка генерации первого вопроса: {e}")
        return "Расскажите подробно о ваших профессиональных навыках и опыте работы?"


async def generate_technical_question(chat_history: list[dict]) -> str:
    """Генерация технического вопроса"""
    candidate_answers = [msg["content"] for msg in chat_history if msg["role"] == "Кандидат"]
    previous_questions = [msg["content"] for msg in chat_history if msg["role"] == "HR"]

    answers_str = "\n".join(candidate_answers)
    questions_str = "\n".join(previous_questions)

    prompt = f"""
    На основе этих ответов кандидата:
    {answers_str}

    И предыдущих вопросов:
    {questions_str}

    Сгенерируй ОДИН технический вопрос, который:
    1. Проверит конкретные навыки, упомянутые кандидатом
    2. Будет достаточно сложным для оценки реальных знаний
    3. Может включать практическое задание или теоретическую задачу
    4. Если кандидат ранее уже сказал, что не обладает каким-то навыком, то не стоит задавать вопросы по данному навыку.
    5. Не повторяй предыдущие вопросы

    Формат: только вопрос в вежливой, официальной форме, без пояснений.
    """
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        logging.error(f"Ошибка генерации технического вопроса: {e}")
        return "Опишите, как бы вы решали задачу обработки больших объемов данных?"


async def determine_final_position(chat_history: list[dict]) -> Position:
    """Определение позиции"""
    conversation = []
    for msg in chat_history:
        if msg["role"] == "HR":
            conversation.append(f"Вопрос: {msg['content']}")
        else:
            conversation.append(f"Ответ: {msg['content']}")
    dialog_str = "\n".join(conversation)

    prompt = f"""
    На основе этого диалога HR с кандидатом:
    {dialog_str}

    Определи, подходит ли кандидат на какую-то из этих позиций (возможно, что у него нет компетенций ни на одну позицию, тогда он нам не подходит):
    {[p.value for p in Position if p != Position.UNSUITABLE]}

    Критерии:
    1. Оценивай только фактические навыки и опыт
    2. Если навыков недостаточно для точного определения позиции (нужны большая часть как общих, так и технических навыков), то он нам не подходит и - верни "{Position.UNSUITABLE.value}"

    Ответь ТОЛЬКО названием позиции или "{Position.UNSUITABLE.value}" (если он не подходит ни на одну из позиций) без пояснений.
    """
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return Position(response.content.strip())
    except Exception as e:
        logging.error(f"Ошибка определения позиции: {e}")
        return Position.UNSUITABLE


async def should_end_early(chat_history: list[dict]) -> tuple[bool, Position]:
    """Проверка возможности досрочного завершения"""
    conversation = []
    for msg in chat_history:
        if msg["role"] == "HR":
            conversation.append(f"Вопрос: {msg['content']}")
        else:
            conversation.append(f"Ответ: {msg['content']}")

    if len(conversation) < 4:
        return False, Position.UNSUITABLE

    dialog_str = "\n".join(conversation)

    prompt = f"""
    На основе этого диалога HR с кандидатом:
    {dialog_str}

    Можно ли УВЕРЕННО и ОДНОЗНАЧНО определить подходящую позицию?
    Доступные позиции: {[p.value for p in Position if p != Position.UNSUITABLE]}
    Если у кандидата нет подходящих навыков ни для одной вакансии (нужно проверить все необходимые технические навыки), то ответь: Нет

    Ответь в формате:
    Да|Нет
    Позиция (если Да)
    """
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        lines = response.content.split('\n')
        if lines[0].strip().lower() == 'да' and len(lines) > 1:
            return True, Position(lines[1].strip())
    except Exception as e:
        logging.error(f"Ошибка проверки досрочного завершения: {e}")
    return False, Position.UNSUITABLE


async def complete_interview(chat_id: int, position: Position):
    """Завершение собеседования"""
    users[chat_id]["state"] = InterviewState.COMPLETED
    if position == Position.UNSUITABLE:
        response = "Спасибо за участие! К сожалению, мы не можем предложить вам подходящую позицию. [Некомпетентный соискатель]"
    else:
        response = f"Спасибо за собеседование! Мы рассмотрим вашу кандидатуру на позицию [{position.value}] и свяжемся с вами."
    await bot.send_message(chat_id, response)


@dp.channel_post(Command("start"))
async def handle_start(message: Message):
    """Инициализация собеседования"""
    users[message.chat.id] = {
        "state": InterviewState.WAITING_FIRST_MESSAGE,
        "position": Position.UNSUITABLE,
        "message_count": 0,
        "chat_history": []}


@dp.channel_post(F.text & ~F.text.startswith('/'))
async def process_message(message: Message):
    """Обработка сообщений кандидата"""
    if message.chat.id not in users:
        return

    user_data = users[message.chat.id]

    if user_data["state"] == InterviewState.WAITING_FIRST_MESSAGE:
        user_data["state"] = InterviewState.IN_PROGRESS
        question = await generate_first_question()
        user_data["chat_history"].append({
            "role": "HR",
            "content": question})

        await bot.send_message(message.chat.id, question)
        return

    if user_data["state"] == InterviewState.IN_PROGRESS:
        user_data["chat_history"].append({
            "role": "Кандидат",
            "content": message.text})

        user_data["message_count"] += 1

        if user_data["message_count"] >= 2 and user_data["message_count"] == 10:
            should_end, position = await should_end_early(user_data["chat_history"])
            if should_end:
                await complete_interview(message.chat.id, position)
                return

        if user_data["message_count"] >= 10:
            final_position = await determine_final_position(user_data["chat_history"])
            await complete_interview(message.chat.id, final_position)
        else:
            question = await generate_technical_question(user_data["chat_history"])
            user_data["chat_history"].append({
                "role": "HR",
                "content": question})

            await bot.send_message(message.chat.id, question)


async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"Ошибка в работе бота: {e}")