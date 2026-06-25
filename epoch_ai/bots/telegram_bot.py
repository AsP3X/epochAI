"""Optional Telegram bot for status and kill-switch control."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from epoch_ai.config.settings import AppConfig
from epoch_ai.execution.kill_switch import KillSwitch
from epoch_ai.services.runtime import RuntimeService
from epoch_ai.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def run_telegram_bot(config: AppConfig) -> None:
    """Start a long-polling Telegram bot (requires python-telegram-bot)."""
    try:
        from telegram import Update
        from telegram.ext import Application, CommandHandler, ContextTypes
    except ImportError as exc:
        raise RuntimeError(
            "python-telegram-bot is required. pip install -r requirements-optional.txt"
        ) from exc

    token = os.environ.get(config.telegram.token_env, "")
    if not token:
        raise RuntimeError(f"Set {config.telegram.token_env} to run the Telegram bot.")

    runtime = RuntimeService(config)
    kill_switch = KillSwitch(config.execution.kill_switch_path)
    allowed = set(config.telegram.allowed_chat_ids)

    def _authorized(chat_id: int) -> bool:
        return not allowed or chat_id in allowed

    async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_chat is None or not _authorized(update.effective_chat.id):
            return
        st = runtime.status()
        ks = kill_switch.read()
        await update.message.reply_text(
            f"epochAI status\n"
            f"symbol: {st.symbol}\n"
            f"model: {st.model_version}\n"
            f"models: {st.models_available}\n"
            f"kill switch: {'HALTED' if ks.halted else 'active'}\n"
            f"reason: {ks.reason or '-'}"
        )

    async def cmd_predict(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_chat is None or not _authorized(update.effective_chat.id):
            return
        from epoch_ai.data.downloader import HistoricalDownloader

        market = HistoricalDownloader(config).load_or_download(config.primary_symbol, n_bars=1200)
        if runtime.status().models_available == 0:
            await update.message.reply_text("No trained models. Run train first.")
            return
        runtime.load_model()
        pred = runtime.predict_market(market)
        await update.message.reply_text(
            f"pred={pred.raw_prediction:.4f} signal={pred.decision.signal} "
            f"conf={pred.decision.confidence:.3f} model={pred.model_version}"
        )

    async def cmd_halt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_chat is None or not _authorized(update.effective_chat.id):
            return
        reason = " ".join(context.args) if context.args else "telegram halt"
        kill_switch.halt(reason)
        await update.message.reply_text(f"Trading halted: {reason}")

    async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_chat is None or not _authorized(update.effective_chat.id):
            return
        kill_switch.resume()
        await update.message.reply_text("Trading resumed.")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("predict", cmd_predict))
    app.add_handler(CommandHandler("halt", cmd_halt))
    app.add_handler(CommandHandler("resume", cmd_resume))
    logger.info("Telegram bot starting (allowed chats=%s)", allowed or "all")
    app.run_polling()
