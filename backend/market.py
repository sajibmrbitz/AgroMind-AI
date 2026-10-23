"""Market metadata and agronomic context for AgroMind AI.

Provides Bengali crop names, seasonal calendar classifications for Bangladesh,
volatility categorization, and utility helpers for user-facing data exports.
"""

from __future__ import annotations
from typing import Dict, Any, List

CROP_METADATA: Dict[str, Dict[str, Any]] = {
    "garlic": {
        "en_name": "Garlic",
        "bn_name": "দেশি রসুন",
        "category": "মশলা জাতীয় ফসল (Spices)",
        "season_bn": "রবি মৌসুম (অক্টোবর - মার্চ)",
        "volatility": "মাঝারি (Moderate)",
        "key_districts": "নাটোর, পাবনা, ফরিদপুর, মেহেরপুর",
        "storage_tip": "পর্যাপ্ত বায়ুচলাচলযুক্ত শুষ্ক মাচায় সংরক্ষণ করলে ২-৩ মাস ভালো থাকে।",
        "unit": "BDT/kg",
        "unit_bn": "টাকা/কেজি",
    },
    "chickpea": {
        "en_name": "Chickpea",
        "bn_name": "ছোলা / বুট",
        "category": "ডাল জাতীয় ফসল (Pulses)",
        "season_bn": "রবি মৌসুম (নভেম্বর - এপ্রিল)",
        "volatility": "কম (Stable)",
        "key_districts": "রাজশাহী, চাঁপাইনবাবগঞ্জ, যশোর, কুষ্টিয়া",
        "storage_tip": "বায়ুরোধী পাত্রে নিমপাতা বা শুকনো মরিচ দিয়ে রাখলে পোকা ধরে না।",
        "unit": "BDT/kg",
        "unit_bn": "টাকা/কেজি",
    },
    "green_chilli": {
        "en_name": "Green Chilli",
        "bn_name": "কাঁচামরিচ",
        "category": "কাঁচা সবজি ও মশলা (Vegetable/Spice)",
        "season_bn": "খরিফ ও রবি (বছরব্যাপী চাষ)",
        "volatility": "অত্যধিক উচ্চ (High Volatility)",
        "key_districts": "বগুড়া, জামালপুর, পঞ্চগড়, চুয়াডাঙ্গা",
        "storage_tip": "পচনশীল ফসল; দ্রুত গ্রেডিং করে ঠান্ডা ও ছায়াযুক্ত স্থানে বাজারজাত করুন।",
        "unit": "BDT/kg",
        "unit_bn": "টাকা/কেজি",
    },
    "cucumber": {
        "en_name": "Cucumber",
        "bn_name": "শসা ও খিরা",
        "category": "সবজি ও ফল (Salad Vegetable)",
        "season_bn": "খরিফ-১ ও রবি (মার্চ - সেপ্টেম্বর / শীতকালীন)",
        "volatility": "উচ্চ (High)",
        "key_districts": "কুমিল্লা, নরসিংদী, যশোর, বগুড়া",
        "storage_tip": "সংরক্ষণ মেয়াদ স্বল্প; তোলা মাত্রই প্যাকেট করে নিকটস্থ মোকামে সরবরাহ জরুরি।",
        "unit": "BDT/kg",
        "unit_bn": "টাকা/কেজি",
    },
    "sweet_pumpkin": {
        "en_name": "Sweet Pumpkin",
        "bn_name": "মিষ্টি কুমড়া",
        "category": "সবজি (Cucurbit Vegetable)",
        "season_bn": "রবি ও খরিফ (নভেম্বর - জুন)",
        "volatility": "কম-মাঝারি (Low-Moderate)",
        "key_districts": "কুড়িগ্রাম, গাইবান্ধা (চরাঞ্চল), রংপুর, ফরিদপুর",
        "storage_tip": "বোঁটা সহ অক্ষত অবস্থায় শুকনো মেঝে বা মাচায় রাখলে ৩-৪ মাস ভালো থাকে।",
        "unit": "BDT/kg",
        "unit_bn": "টাকা/কেজি",
    },
}


def get_crop_info(crop: str) -> Dict[str, Any]:
    """Return agricultural metadata for a given crop identifier."""
    if crop not in CROP_METADATA:
        raise ValueError(f"Unknown crop: '{crop}'. Available crops: {list(CROP_METADATA.keys())}")
    return CROP_METADATA[crop]


def get_crop_display_label(crop: str) -> str:
    """Return formatted bilingual display label for dropdown selection."""
    info = get_crop_info(crop)
    return f"{info['bn_name']} ({info['en_name']})"


def list_available_crops() -> List[str]:
    """Return list of supported crop keys."""
    return list(CROP_METADATA.keys())
