"""TelegramNotifier - Send prediction alerts via Telegram.

Handles formatting and delivery of prediction signals to Telegram.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send prediction alerts via Telegram bot.

    Usage:
        notifier = TelegramNotifier(bot_token="...", chat_id="...")
        await notifier.send_signal(signal)

    To get bot_token and chat_id:
    1. Create a bot via @BotFather on Telegram
    2. Get your chat_id by messaging the bot and checking the API
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        """Initialize TelegramNotifier.

        Args:
            bot_token: Telegram bot token from BotFather
            chat_id: Chat ID to send messages to
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)

        if not self._enabled:
            logger.warning("Telegram not configured - notifications disabled")

    async def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Send a message via Telegram.

        Args:
            text: Message text
            parse_mode: Parse mode (Markdown or HTML)

        Returns:
            True if successful
        """
        if not self._enabled:
            logger.info(f"[Telegram disabled] Would send: {text[:100]}...")
            return False

        try:
            import aiohttp
        except ImportError:
            return await self._send_message_sync(text, parse_mode)

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Telegram message sent successfully")
                        return True
                    else:
                        error = await response.text()
                        logger.error(f"Telegram API error: {response.status} - {error}")
                        return False
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    async def _send_message_sync(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Sync fallback for sending messages."""
        import requests

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("Telegram message sent successfully")
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    async def send_signal(self, signal) -> bool:
        """Send a PredictionSignal as a formatted message.

        Args:
            signal: PredictionSignal object

        Returns:
            True if successful
        """
        message = signal.to_telegram_message()
        return await self.send_message(message)

    async def send_hourly_summary(self, results: dict) -> bool:
        """Send hourly performance summary.

        Args:
            results: Dict with performance metrics

        Returns:
            True if successful
        """
        lines = [
            "📊 **Hourly Performance Summary**",
            "",
        ]

        for symbol, metrics in results.items():
            if metrics:
                direction_correct = metrics.get("direction_correct", False)
                emoji = "✅" if direction_correct else "❌"
                actual_return = metrics.get("actual_return_pct", 0)
                predicted_return = metrics.get("predicted_return_pct", 0)

                lines.extend([
                    f"{emoji} **{symbol}**",
                    f"   Predicted: {predicted_return:+.2f}%",
                    f"   Actual: {actual_return:+.2f}%",
                    f"   Error: {abs(actual_return - predicted_return):.2f}%",
                    "",
                ])

        # Overall stats
        total = len(results)
        correct = sum(1 for m in results.values() if m and m.get("direction_correct", False))
        accuracy = correct / total * 100 if total > 0 else 0

        lines.extend([
            f"**Accuracy:** {correct}/{total} ({accuracy:.0f}%)",
        ])

        message = "\n".join(lines)
        return await self.send_message(message)

    async def send_adaptation_update(self, old_params: dict, new_params: dict) -> bool:
        """Send notification about parameter adaptation.

        Args:
            old_params: Previous parameters
            new_params: New parameters

        Returns:
            True if successful
        """
        lines = [
            "🔧 **ML Adaptation Update**",
            "",
        ]

        for key in new_params:
            old_val = old_params.get(key, "N/A")
            new_val = new_params[key]
            if old_val != new_val:
                lines.append(f"• {key}: {old_val} → {new_val}")

        if len(lines) == 2:
            lines.append("No parameters changed")

        message = "\n".join(lines)
        return await self.send_message(message)

    def configure(self, bot_token: str, chat_id: str):
        """Configure Telegram credentials.

        Args:
            bot_token: Bot token from BotFather
            chat_id: Chat ID to send to
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._enabled = True
        logger.info("Telegram notifier configured")

    @property
    def is_enabled(self) -> bool:
        """Check if Telegram is configured and enabled."""
        return self._enabled
