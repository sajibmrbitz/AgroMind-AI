"""Market metadata and agronomic context for AgroMind AI.

Provides formal Bengali crop classifications, seasonal calendars, volatility metrics,
and agronomic guidance aligned with standard institutional terminology.
"""

from __future__ import annotations
from typing import Dict, Any, List

CROP_METADATA: Dict[str, Dict[str, Any]] = {
    "garlic": {
        "en_name": "Garlic",
        "bn_name": "দেশি রসুন",
        "category": "মশলা জাতীয় ফসল (Spices)",
        "season_bn": "রবি মৌসুম (অক্টোবর - মার্চ)",
        "volatility": "মাঝারি অস্থিরতা (Moderate Volatility)",
        "key_districts": "নাটোর, পাবনা, ফরিদপুর, মেহেরপুর",
        "storage_tip": "পর্যাপ্ত বায়ুচলাচলযুক্ত শুষ্ক মাচা বা ঝুলন্ত জালে সংরক্ষণ করলে আর্দ্রতাজনিত পচন রোধ করা সম্ভব এবং ২-৩ মাস গুণমান অক্ষুণ্ণ থাকে।",
        "unit": "BDT/kg",
        "unit_bn": "টাকা/কেজি",
    },
    "chickpea": {
        "en_name": "Chickpea",
        "bn_name": "ছোলা / বুট",
        "category": "ডাল জাতীয় ফসল (Pulses)",
        "season_bn": "রবি মৌসুম (নভেম্বর - এপ্রিল)",
        "volatility": "স্থিতিশীল (Stable Commodity)",
        "key_districts": "রাজশাহী, চাঁপাইনবাবগঞ্জ, যশোর, কুষ্টিয়া",
        "storage_tip": "আর্দ্রতা ১০% এর নিচে নামিয়ে বায়ুরোধী সাইলো বা পাত্রে সংরক্ষণ করলে গুণমান ও পুষ্টিমান দীর্ঘদিন সুরক্ষিত থাকে।",
        "unit": "BDT/kg",
        "unit_bn": "টাকা/কেজি",
    },
    "green_chilli": {
        "en_name": "Green Chilli",
        "bn_name": "কাঁচামরিচ",
        "category": "কাঁচা সবজি ও মশলা (Vegetable/Spice)",
        "season_bn": "খরিফ ও রবি (বছরব্যাপী চাষাবাদ)",
        "volatility": "অত্যধিক উচ্চ অস্থিরতা (High Volatility)",
        "key_districts": "বগুড়া, জামালপুর, পঞ্চগড়, চুয়াডাঙ্গা",
        "storage_tip": "উচ্চমাত্রায় পচনশীল বিধায় ফসল সংগ্রহের পর সরাসরি সূর্যালোক পরিহার করে গ্রেডিং শেষে দ্রুত নিকটস্থ পাইকারি মোকামে প্রেরণ আবশ্যক।",
        "unit": "BDT/kg",
        "unit_bn": "টাকা/কেজি",
    },
    "cucumber": {
        "en_name": "Cucumber",
        "bn_name": "শসা ও খিরা",
        "category": "ফলজাতীয় সবজি (Cucurbit Vegetable)",
        "season_bn": "খরিফ-১ ও রবি (মার্চ - সেপ্টেম্বর / শীতকালীন)",
        "volatility": "উচ্চ অস্থিরতা (High Volatility)",
        "key_districts": "কুমিল্লা, নরসিংদী, যশোর, বগুড়া",
        "storage_tip": "সংরক্ষণ স্থায়িত্ব সীমিত বিধায় সংগ্রহের পর আর্দ্র-শীতল পরিবেশে রেখে স্বল্পতম সময়ে আড়ত ও বিপণন কেন্দ্রে সরবরাহ করা সমীচীন।",
        "unit": "BDT/kg",
        "unit_bn": "টাকা/কেজি",
    },
    "sweet_pumpkin": {
        "en_name": "Sweet Pumpkin",
        "bn_name": "মিষ্টি কুমড়া",
        "category": "কন্দ ও লতাজাতীয় সবজি (Cucurbits)",
        "season_bn": "রবি ও খরিফ (নভেম্বর - জুন)",
        "volatility": "স্বল্প-মাঝারি অস্থিরতা (Low-Moderate Volatility)",
        "key_districts": "কুড়িগ্রাম, গাইবান্ধা (চরাঞ্চল), রংপুর, ফরিদপুর",
        "storage_tip": "বোঁটাসহ পরিপক্ব ফল সংগ্রহ করে শুষ্ক, ছায়াযুক্ত মেঝ বা বাঁশের মাচায় স্তূপাকারে রাখলে ৩-৪ মাস স্বাভাবিক গুণমান বজায় থাকে।",
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
