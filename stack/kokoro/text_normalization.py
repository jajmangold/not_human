"""Conservative semantic text normalization before Kokoro/Misaki G2P."""

from __future__ import annotations

import calendar
import re
import unicodedata
from decimal import Decimal

from num2words import num2words


_DIGIT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_UNITS = {
    "hz": ("hertz", "hertz"), "khz": ("kilohertz", "kilohertz"),
    "mhz": ("megahertz", "megahertz"), "ghz": ("gigahertz", "gigahertz"),
    "kb": ("kilobyte", "kilobytes"), "mb": ("megabyte", "megabytes"),
    "gb": ("gigabyte", "gigabytes"), "tb": ("terabyte", "terabytes"),
    "ms": ("millisecond", "milliseconds"), "s": ("second", "seconds"),
    "kg": ("kilogram", "kilograms"), "g": ("gram", "grams"),
    "km": ("kilometer", "kilometers"), "cm": ("centimeter", "centimeters"),
    "mm": ("millimeter", "millimeters"), "mph": ("mile per hour", "miles per hour"),
    "fps": ("frame per second", "frames per second"),
}

_AUDIO_CUE = re.compile(
    r"\[(?:laugh(?:ter)?|chuckle|giggle|sigh|whisper|cough|cry|sob|gasp|"
    r"breath(?:e)?|sniff|music|sound\s+effect)\]",
    re.I,
)
_PARENTHETICAL_CUE = re.compile(
    r"\((?:laughs?|chuckles?|giggles?|sighs?|whispers?|coughs?|cries?|"
    r"sobs?|gasps?|breathes?|sniffs?|sound\s+effects?)\)",
    re.I,
)


def _cardinal(value: str | int) -> str:
    return num2words(int(str(value).replace(",", "")), lang="en").replace(" and ", " ")


def _digits(value: str) -> str:
    return " ".join(_DIGIT_WORDS[int(char)] for char in value if char.isdigit())


def _phone(match: re.Match[str]) -> str:
    raw = match.group(0)
    extension = re.search(r"(?:ext\.?|x)\s*(\d+)$", raw, re.I)
    main = raw[:extension.start()] if extension else raw
    digits = re.sub(r"\D", "", main)
    country = ""
    if main.lstrip().startswith("+") and len(digits) > 10:
        country_digits, digits = digits[:-10], digits[-10:]
        country = f"plus {_digits(country_digits)}, "
    if len(digits) == 10:
        spoken = f"{_digits(digits[:3])}, {_digits(digits[3:6])}, {_digits(digits[6:])}"
    else:
        spoken = _digits(digits)
    if extension:
        spoken += f", extension {_digits(extension.group(1))}"
    return country + spoken


def _currency(match: re.Match[str]) -> str:
    sign, symbol, raw = match.group(1), match.group(2), match.group(3).replace(",", "")
    names = {"$": ("dollar", "cent"), "£": ("pound", "penny"), "€": ("euro", "cent")}
    major_name, minor_name = names[symbol]
    value = Decimal(raw)
    major, minor = int(value), int((value - int(value)) * 100)
    result = f"{_cardinal(major)} {major_name if major == 1 else major_name + 's'}"
    if minor:
        minor_plural = "pence" if symbol == "£" else minor_name + ("" if minor == 1 else "s")
        result += f" and {_cardinal(minor)} {minor_plural}"
    return ("negative " if sign else "") + result


def _iso_date(match: re.Match[str]) -> str:
    year, month, day = map(int, match.groups())
    try:
        calendar.monthrange(year, month)[1]
        if day < 1 or day > calendar.monthrange(year, month)[1]:
            return match.group(0)
    except (calendar.IllegalMonthError, ValueError):
        return match.group(0)
    return f"{calendar.month_name[month]} {num2words(day, to='ordinal')}, {num2words(year, to='year')}"


def _time(match: re.Match[str]) -> str:
    hour, minute, suffix = int(match.group(1)), int(match.group(2)), match.group(3)
    if hour > 23 or minute > 59 or (suffix and not 1 <= hour <= 12):
        return match.group(0)
    minute_words = "o'clock" if minute == 0 else (f"oh {_cardinal(minute)}" if minute < 10 else _cardinal(minute))
    spoken = f"{_cardinal(hour)} {minute_words}"
    trailing_period = "." if match.group(0).rstrip().endswith(".") else ""
    if suffix:
        spoken += " " + " ".join(suffix.upper().replace(".", ""))
    return spoken + trailing_period


def _strip_speech_markup(text: str) -> str:
    """Remove presentation syntax that Misaki would otherwise pronounce.

    Kokoro accepts ordinary punctuation as its pacing control, but it does not
    understand Markdown, SSML, or stage directions.  In particular, an asterisk
    left by emphasis or a list marker is tokenized as the word "asterisk".  This
    cleanup is deliberately limited to presentation artifacts; the transcript
    shown in the browser remains the model's original text.
    """
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"<[^>\n]{1,160}>", " ", text)
    text = _AUDIO_CUE.sub(" ", text)
    text = _PARENTHETICAL_CUE.sub(" ", text)

    # Fenced code and links are already handled by normalize_for_speech.  These
    # rules remove the remaining Markdown surface syntax without touching words.
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    text = re.sub(r"(?m)^\s{0,3}(?:[*+\-]|\d+[.)])\s+", "", text)
    text = re.sub(r"(?m)^\s*(?:\*{3,}|_{3,}|-{3,})\s*$", " ", text)
    text = re.sub(r"(?<!\w)\*{1,3}(?=\S)", "", text)
    text = re.sub(r"(?<=\S)\*{1,3}(?!\w)", "", text)
    text = re.sub(r"(?<!\w)_{1,3}(?=\S)", "", text)
    text = re.sub(r"(?<=\S)_{1,3}(?!\w)", "", text)
    text = re.sub(r"(?<!\w)~{1,2}(?=\S)|(?<=\S)~{1,2}(?!\w)", "", text)

    # Preserve a useful spoken form for simple multiplication before dropping
    # any remaining standalone decorative symbols.
    text = re.sub(
        r"(?<!\w)(-?\d+(?:\.\d+)?)\s*\*\s*(-?\d+(?:\.\d+)?)(?!\w)",
        r"\1 times \2",
        text,
    )
    text = re.sub(r"(?<=\w)\s+\*\s+(?=\w)", " times ", text)
    text = re.sub(r"(?<!\w)\*+(?!\w)", " ", text)
    text = re.sub(r"(?<!\w)\|+(?!\w)|\\+(?=\S)|\^+(?=\S)", " ", text)
    return text


def normalize_for_speech(text: str) -> str:
    """Expand only formats whose spoken meaning is sufficiently unambiguous."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\[([^]]+)]\([^)]*\)", r"\1", text)
    text = text.replace("`", "")
    text = _strip_speech_markup(text)
    text = re.sub(
        r"(?<![\w-])(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}"
        r"(?:\s*(?:ext\.?|x)\s*\d+)?(?!\w)",
        _phone,
        text,
        flags=re.I,
    )
    text = re.sub(r"(?<!\w)(\d{4})-(\d{2})-(\d{2})(?!\w)", _iso_date, text)
    text = re.sub(r"(?<!\w)(\d{1,2}):(\d{2})\s*([ap]\.?m\.?)?(?!\w)", _time, text, flags=re.I)
    text = re.sub(
        r"(-?)([$£€])((?:\d{1,3}(?:,\d{3}){1,4}|\d{1,15})(?:\.\d{1,2})?)(?!\d|\.\d)",
        _currency,
        text,
    )
    text = re.sub(
        r"(?<!\w)(\d{1,12}(?:\.\d{1,4})?)%",
        lambda match: f"{num2words(Decimal(match.group(1)))} percent",
        text,
    )

    def unit(match: re.Match[str]) -> str:
        raw, suffix = match.group(1), match.group(2).lower()
        amount = Decimal(raw)
        name = _UNITS[suffix][0 if amount == 1 else 1]
        return f"{num2words(amount)} {name}"

    unit_names = "|".join(sorted(_UNITS, key=len, reverse=True))
    text = re.sub(rf"(?<!\w)(\d{{1,12}}(?:\.\d{{1,4}})?)\s*({unit_names})(?!\w)", unit, text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text
