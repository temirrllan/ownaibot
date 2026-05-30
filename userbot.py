#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram userbot с ИИ-фильтром новостей.

Работает от имени личного аккаунта (MTProto / Telethon), читает все каналы
из ленты, прогоняет сообщения через LLM (OpenAI) и пересылает подходящие
в Saved Messages (Избранное).

ВНИМАНИЕ ПО БЕЗОПАСНОСТИ:
  Файл сессии (*.session) = полный доступ к вашему аккаунту Telegram.
  Никому его не передавайте, не выкладывайте в репозиторий/облако.
  Ключи (API_ID/API_HASH/OPENAI_KEY) держите в переменных окружения
  или в файле .env (см. .env.example), а НЕ в коде.
"""

import asyncio
import hashlib
import logging
import os
import re
import sys
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

from telethon import TelegramClient, events
from telethon.tl.types import Channel, MessageMediaWebPage
from telethon.errors import FloodWaitError
from openai import AsyncOpenAI, OpenAIError

# ---------------------------------------------------------------------------
# Загрузка .env (опционально). Если установлен python-dotenv — подхватим .env.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # без dotenv просто берём переменные окружения как есть


# ===========================================================================
#                              НАСТРОЙКИ
#  Чувствительные значения берём из переменных окружения (см. .env.example).
#  Остальное можно править прямо здесь.
# ===========================================================================

# --- Авторизация Telegram (с https://my.telegram.org) ---
API_ID = int(os.getenv("API_ID", "0"))          # ваш api_id (число)
API_HASH = os.getenv("API_HASH", "")            # ваш api_hash (строка)

# --- OpenAI ---
OPENAI_KEY = os.getenv("OPENAI_KEY", "")        # ключ OpenAI
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- Управляющий бот (токен от @BotFather). Если пусто — бот не запускается,
#     мониторинг включается автоматически при старте процесса. ---
BOT_TOKEN = os.getenv("BOT_TOKEN", "")

# --- Имя файла сессии Telethon (создаётся при первом входе) ---
SESSION_NAME = os.getenv("SESSION_NAME", "userbot")

# --- Куда пересылать отобранные сообщения. "me" = Избранное ---
TARGET = "me"

# --- Параметры стартовой сводки (backfill) ---
BACKFILL_HOURS = int(os.getenv("BACKFILL_HOURS", "24"))   # за сколько часов назад
BACKFILL_LIMIT_PER_CHANNEL = int(os.getenv("BACKFILL_LIMIT_PER_CHANNEL", "50"))

# --- Минимальная длина сообщения (короче — не отправляем в ИИ) ---
MIN_LEN = int(os.getenv("MIN_LEN", "60"))

# --- Дедупликация ---
DEDUP_MAX = int(os.getenv("DEDUP_MAX", "5000"))   # максимум хранимых отпечатков

# --- Задержка между запросами в backfill, чтобы беречь лимиты Telegram/OpenAI ---
BACKFILL_DELAY_SEC = float(os.getenv("BACKFILL_DELAY_SEC", "0.7"))

# --- Описание интересов (правьте свободно своими словами) ---
MY_INTERESTS = os.getenv("MY_INTERESTS", """
Я разработчик. Отбирай посты, которые реально полезны и связаны с темами ниже.

ИНТЕРЕСНО (отбирать):
- Искусственный интеллект и ML: ВЫХОД НОВЫХ ИИ-МОДЕЛЕЙ и их анонсы
  (GPT, Claude, Gemini, Llama, и любые другие новые модели/нейросети),
  LLM, AI-инструменты и сервисы, AI-агенты, промпт-инжиниринг,
  исследования и крупные новости в области AI.
- Claude и его экосистема: Claude Code, СКИЛЛЫ ДЛЯ CLAUDE (Claude skills),
  MCP (Model Context Protocol) и MCP-серверы, КОННЕКТОРЫ для Claude,
  интеграции, Anthropic, обновления и возможности Claude, гайды и приёмы.
- Кодинг и разработка: программирование, языки и фреймворки, backend,
  Python, базы данных, DevOps, архитектура, инструменты разработчика,
  полезные технические статьи, гайды, разборы, крупные релизы и обновления.
- Автоматизация: скрипты, боты, пайплайны, no-code/low-code, интеграции,
  автоматизация рабочих процессов.
- IT в целом: значимые новости индустрии, новые технологии и продукты,
  стартапы, технологические тренды.
- Вакансии в IT: разработка, backend, Python, ML/AI, дата (удалёнка или
  релокация особенно ценны).
- Мероприятия: митапы, конференции, хакатоны, вебинары, воркшопы по IT/AI.
- Влиятельные люди в IT/AI: интервью, мнения, новости про заметных
  основателей, инженеров, исследователей и лидеров индустрии.
- Развитие Instagram и личный бренд: продвижение и рост в Instagram,
  как делать КАРУСЕЛИ и контент, охваты и алгоритмы, упаковка профиля,
  продвижение себя как специалиста, личный бренд, создание контента,
  привлечение аудитории (особенно в IT/AI/digital-нише).

НЕ ИНТЕРЕСНО (отклонять):
- Реклама курсов, инфоцыганство, «успешный успех», обещания лёгких денег.
- Розыгрыши, конкурсы, giveaway, накрутки, реферальные ссылки.
- Мемы, развлекательный оффтоп, гороскопы, новости не по теме (политика,
  спорт, шоу-бизнес), если они не касаются тем выше.
- Чисто рекламные посты и платные интеграции без полезной сути.

Если сомневаешься и пост не относится к темам выше — отклоняй.
""").strip()

# --- Рабочие файлы состояния (лежат рядом со скриптом) ---
BASE_DIR = Path(__file__).resolve().parent
DEDUP_FILE = BASE_DIR / "dedup.txt"               # отпечатки уже пересланных новостей
BACKFILL_MARKER = BASE_DIR / "backfill.done"      # маркер выполненной сводки
STATE_FILE = BASE_DIR / "monitor.state"           # вкл/выкл мониторинг (пульт-бот)


# ===========================================================================
#                              ЛОГИРОВАНИЕ
# ===========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("userbot")
# Telethon очень болтлив на DEBUG — оставляем только предупреждения.
logging.getLogger("telethon").setLevel(logging.WARNING)


# ===========================================================================
#                       ДЕДУПЛИКАЦИЯ (отпечатки текста)
# ===========================================================================
class Dedup:
    """
    Хранит хэши нормализованных текстов уже пересланных новостей.
    Нормализация: убираем ссылки, пунктуацию, лишние пробелы, нижний регистр.
    Это позволяет ловить одинаковые новости из разных каналов.
    Файл ограничен по размеру (DEDUP_MAX строк), старые записи вытесняются.
    """

    _url_re = re.compile(r"https?://\S+|t\.me/\S+|@\w+")
    _non_word_re = re.compile(r"[^\w\s]", re.UNICODE)
    _space_re = re.compile(r"\s+")

    def __init__(self, path: Path, max_size: int):
        self.path = path
        self.max_size = max_size
        # deque сохраняет порядок и позволяет вытеснять самые старые записи
        self._hashes: deque[str] = deque(maxlen=max_size)
        self._set: set[str] = set()
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                h = line.strip()
                if h:
                    self._hashes.append(h)
            self._set = set(self._hashes)
            log.info("Загружено отпечатков дедупа: %d", len(self._set))

    @classmethod
    def fingerprint(cls, text: str) -> str:
        """Возвращает стабильный хэш нормализованного текста."""
        t = text.lower()
        t = cls._url_re.sub(" ", t)
        t = cls._non_word_re.sub(" ", t)
        t = cls._space_re.sub(" ", t).strip()
        return hashlib.sha256(t.encode("utf-8")).hexdigest()

    def seen(self, text: str) -> bool:
        return self.fingerprint(text) in self._set

    def add(self, text: str) -> None:
        h = self.fingerprint(text)
        if h in self._set:
            return
        # если deque переполнен — самый старый элемент вытеснится автоматически,
        # нам нужно синхронно убрать его из set
        if len(self._hashes) == self._hashes.maxlen:
            oldest = self._hashes[0]
            self._set.discard(oldest)
        self._hashes.append(h)
        self._set.add(h)
        self._persist()

    def _persist(self) -> None:
        # пишем атомарно: сначала во временный файл, потом заменяем
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("\n".join(self._hashes), encoding="utf-8")
        tmp.replace(self.path)


# ===========================================================================
#               СОСТОЯНИЕ МОНИТОРИНГА (управляется ботом /start /stop)
# ===========================================================================
class MonitorState:
    """
    Хранит флаг «мониторить или нет» и сохраняет его в файл,
    чтобы после рестарта процесса состояние не сбрасывалось.
    """

    def __init__(self, path: Path, default: bool):
        self.path = path
        if path.exists():
            self.enabled = path.read_text(encoding="utf-8").strip() == "on"
        else:
            self.enabled = default

    def set(self, value: bool) -> None:
        self.enabled = value
        self.path.write_text("on" if value else "off", encoding="utf-8")


# ===========================================================================
#                          ИИ-ФИЛЬТР (OpenAI)
# ===========================================================================
class AIFilter:
    """Решает, интересно сообщение или нет, на основе MY_INTERESTS."""

    SYSTEM_PROMPT = (
        "Ты — персональный фильтр новостей. На основе описания интересов "
        "пользователя реши, стоит ли показывать ему данное сообщение из "
        "Telegram-канала. Отвечай СТРОГО в формате одной строки:\n"
        "RELEVANT|<короткая причина по-русски, до 8 слов>\n"
        "или\n"
        "SKIP|<короткая причина>\n"
        "Никакого другого текста. Будь строг: рекламу, инфоцыганство, "
        "розыгрыши и мемы отклоняй."
    )

    def __init__(self, client: AsyncOpenAI, model: str, interests: str):
        self.client = client
        self.model = model
        self.interests = interests

    async def judge(self, text: str) -> tuple[bool, str]:
        """
        Возвращает (relevant, reason).
        При ошибке API считаем сообщение НЕ релевантным (тихо пропускаем),
        чтобы не спамить Избранное и не падать.
        """
        user_prompt = (
            f"ИНТЕРЕСЫ ПОЛЬЗОВАТЕЛЯ:\n{self.interests}\n\n"
            f"СООБЩЕНИЕ ИЗ КАНАЛА:\n{text[:4000]}"
        )
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=40,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            answer = (resp.choices[0].message.content or "").strip()
        except OpenAIError as e:
            log.warning("Ошибка OpenAI: %s", e)
            return False, "ошибка ИИ"

        relevant = answer.upper().startswith("RELEVANT")
        reason = answer.split("|", 1)[1].strip() if "|" in answer else answer
        return relevant, reason[:120]


# ===========================================================================
#                       ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ===========================================================================
def is_broadcast_channel(chat) -> bool:
    """True только для каналов-broadcast (не группы, не личка)."""
    return isinstance(chat, Channel) and getattr(chat, "broadcast", False)


def channel_title(chat) -> str:
    return getattr(chat, "title", None) or getattr(chat, "username", None) or "канал"


def message_link(chat, message) -> str | None:
    """
    Возвращает ссылку на конкретный пост.
    Публичный канал: https://t.me/username/123
    Приватный канал: https://t.me/c/<id>/123 (откроется, если ты подписан)
    """
    username = getattr(chat, "username", None)
    if username:
        return f"https://t.me/{username}/{message.id}"
    cid = getattr(chat, "id", None)
    if cid:
        return f"https://t.me/c/{cid}/{message.id}"
    return None


async def process_message(client, ai: AIFilter, dedup: Dedup, message, chat) -> bool:
    """
    Общая логика обработки одного сообщения (используется и в backfill,
    и в реальном времени). Возвращает True, если сообщение переслано.
    """
    text = message.message or ""  # текст сообщения (без медиа-подписей сложных типов)
    if len(text.strip()) < MIN_LEN:
        log.info("⏭ Коротко, в ИИ не шлём (%s): %d симв.",
                 channel_title(chat), len(text.strip()))
        return False  # слишком короткое — не тратим запрос к ИИ

    if dedup.seen(text):
        log.info("Дубль пропущен (%s)", channel_title(chat))
        return False

    relevant, reason = await ai.judge(text)
    if not relevant:
        log.info("⏭ ИИ отклонил (%s): %s", channel_title(chat), reason)
        return False

    title = channel_title(chat)
    header = f"📰 Из «{title}»\n🤖 Причина: {reason}"
    link = message_link(chat, message)
    link_line = f"\n\n🔗 {link}" if link else ""

    # есть ли у поста настоящее медиа (фото/видео/документ), а не просто
    # превью ссылки. Такое медиа можно отправить копией.
    media = message.media
    has_media = media is not None and not isinstance(media, MessageMediaWebPage)

    def _trim(body: str, room: int) -> str:
        body = body.strip()
        if len(body) > room:
            body = body[:room].rstrip() + "…"
        return body

    async def _deliver():
        # Отправляем КОПИЮ поста (а не пересылку): обычное новое сообщение
        # стабильно видно на телефоне и не зависит от настроек канала.
        if has_media:
            # подпись у медиа ограничена ~1024 символами
            body = _trim(text, 1024 - len(header) - len(link_line) - 4)
            caption = f"{header}\n\n{body}{link_line}" if body else f"{header}{link_line}"
            try:
                await client.send_file(TARGET, media, caption=caption)
                return
            except Exception as e:
                log.info("Медиа не отправилось (%s): %s — шлём текстом", title, e)
        # текстовый вариант (или фолбэк, если медиа не ушло): до ~4096 символов
        body = _trim(text, 3900 - len(header) - len(link_line))
        msg = f"{header}\n\n{body}{link_line}" if body else f"{header}{link_line}"
        await client.send_message(TARGET, msg, link_preview=False)

    try:
        await _deliver()
    except FloodWaitError as e:
        log.warning("FloodWait %d сек — ждём", e.seconds)
        await asyncio.sleep(e.seconds + 1)
        await _deliver()

    dedup.add(text)
    log.info("✅ Отправлено из «%s»: %s", title, reason)
    return True


# ===========================================================================
#                       СТАРТОВАЯ СВОДКА (BACKFILL)
# ===========================================================================
async def run_backfill(client, ai: AIFilter, dedup: Dedup, state) -> None:
    if BACKFILL_MARKER.exists():
        log.info("Маркер сводки найден — backfill пропускаем.")
        return

    log.info("Старт сводки за последние %d ч...", BACKFILL_HOURS)
    since = datetime.now(timezone.utc) - timedelta(hours=BACKFILL_HOURS)
    forwarded = 0

    async for dialog in client.iter_dialogs():
        if not state.enabled:
            log.info("Сводка прервана командой /stop.")
            return  # маркер НЕ ставим — при следующем /start сводка продолжится
        chat = dialog.entity
        if not is_broadcast_channel(chat):
            continue

        log.info("Сводка по каналу: %s", channel_title(chat))
        count = 0
        try:
            async for message in client.iter_messages(
                chat, limit=BACKFILL_LIMIT_PER_CHANNEL
            ):
                if not state.enabled:
                    log.info("Сводка прервана командой /stop.")
                    return
                if message.date and message.date < since:
                    break  # сообщения идут от новых к старым — дальше только старее
                count += 1
                if await process_message(client, ai, dedup, message, chat):
                    forwarded += 1
                await asyncio.sleep(BACKFILL_DELAY_SEC)  # бережём лимиты
        except FloodWaitError as e:
            log.warning("FloodWait %d сек на канале %s", e.seconds, channel_title(chat))
            await asyncio.sleep(e.seconds + 1)
        except Exception as e:  # один битый канал не должен ронять всю сводку
            log.warning("Ошибка на канале %s: %s", channel_title(chat), e)

    BACKFILL_MARKER.write_text(
        f"done at {datetime.now(timezone.utc).isoformat()}\n", encoding="utf-8"
    )
    log.info("Сводка завершена. Переслано: %d. Маркер установлен.", forwarded)


# ===========================================================================
#                              ТОЧКА ВХОДА
# ===========================================================================
async def main() -> None:
    # --- проверка обязательных настроек ---
    if not API_ID or not API_HASH:
        log.error("Не заданы API_ID / API_HASH (см. .env.example).")
        sys.exit(1)
    if not OPENAI_KEY:
        log.error("Не задан OPENAI_KEY (см. .env.example).")
        sys.exit(1)

    ai = AIFilter(AsyncOpenAI(api_key=OPENAI_KEY), OPENAI_MODEL, MY_INTERESTS)
    dedup = Dedup(DEDUP_FILE, DEDUP_MAX)

    # Если задан BOT_TOKEN — мониторинг по умолчанию ВЫКЛ (ждём /start от пульта).
    # Если бота нет — мониторим сразу (поведение как раньше).
    state = MonitorState(STATE_FILE, default=not bool(BOT_TOKEN))

    # session — файл рядом со скриптом, чтобы при рестарте не логиниться заново
    client = TelegramClient(str(BASE_DIR / SESSION_NAME), API_ID, API_HASH)

    # --- обработчик новых сообщений в реальном времени (userbot) ---
    @client.on(events.NewMessage)
    async def handler(event):
        if not state.enabled:
            return  # мониторинг выключен командой /stop
        try:
            chat = await event.get_chat()
            if not is_broadcast_channel(chat):
                return  # только broadcast-каналы
            await process_message(client, ai, dedup, event.message, chat)
        except FloodWaitError as e:
            log.warning("FloodWait %d сек", e.seconds)
            await asyncio.sleep(e.seconds + 1)
        except Exception as e:
            log.warning("Ошибка обработки сообщения: %s", e)

    await client.start()  # при первом запуске спросит телефон + код из Telegram
    me = await client.get_me()
    log.info("Вошли как: %s (id=%s)", me.first_name, me.id)

    # backfill запускаем в фоне, чтобы он не блокировал приём команд бота
    async def maybe_backfill():
        try:
            await run_backfill(client, ai, dedup, state)
        except Exception as e:
            log.warning("Ошибка backfill: %s", e)

    # --- управляющий бот (пульт): /start /stop /status ---
    if BOT_TOKEN:
        bot = TelegramClient(str(BASE_DIR / "control_bot"), API_ID, API_HASH)

        def is_owner(event) -> bool:
            # командам подчиняемся только от владельца userbot (тот же человек)
            return event.sender_id == me.id

        @bot.on(events.NewMessage(pattern=r"/start"))
        async def cmd_start(event):
            if not is_owner(event):
                return
            if state.enabled:
                await event.respond("▶️ Мониторинг уже включён.")
                return
            state.set(True)
            await event.respond(
                "▶️ Мониторинг включён. Читаю каналы и шлю подходящее в «Избранное»."
            )
            log.info("Мониторинг ВКЛючён по команде.")
            asyncio.create_task(maybe_backfill())  # стартовая сводка, если ещё не было

        @bot.on(events.NewMessage(pattern=r"/stop"))
        async def cmd_stop(event):
            if not is_owner(event):
                return
            state.set(False)
            await event.respond("⏹️ Мониторинг остановлен. /start — чтобы включить снова.")
            log.info("Мониторинг ВЫКЛючён по команде.")

        @bot.on(events.NewMessage(pattern=r"/status"))
        async def cmd_status(event):
            if not is_owner(event):
                return
            s = "включён ▶️" if state.enabled else "выключен ⏹️"
            await event.respond(f"Статус мониторинга: {s}")

        await bot.start(bot_token=BOT_TOKEN)
        bot_me = await bot.get_me()
        log.info("Пульт-бот запущен: @%s. Команды: /start /stop /status", bot_me.username)

        # если после рестарта мониторинг был включён — досводим (если не было сводки)
        if state.enabled:
            asyncio.create_task(maybe_backfill())

        log.info("Жду команды от бота и слушаю каналы...")
        await asyncio.gather(
            client.run_until_disconnected(),
            bot.run_until_disconnected(),
        )
    else:
        # режим без бота: сразу сводка + слушаем каналы (старое поведение)
        await maybe_backfill()
        log.info("Слушаю новые сообщения в реальном времени... (Ctrl+C для выхода)")
        await client.run_until_disconnected()  # сам реконнектится при обрывах


if __name__ == "__main__":
    # Внешний цикл перезапуска: если выпадет необработанное исключение —
    # подождём и попробуем снова (на VPS systemd тоже подстрахует Restart=always).
    while True:
        try:
            asyncio.run(main())
            break  # штатное завершение (например, разлогин) — выходим
        except KeyboardInterrupt:
            log.info("Остановлено пользователем.")
            break
        except Exception as e:
            log.error("Критическая ошибка: %s. Перезапуск через 15 сек...", e)
            time.sleep(15)
