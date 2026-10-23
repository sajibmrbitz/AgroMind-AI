"""Market metadata and agronomic context for AgroMind AI.

Provides bilingual (Bengali & English) crop classifications, seasonal calendars,
volatility metrics, and agronomic guidance aligned with institutional standards.
"""

from __future__ import annotations
from typing import Dict, Any, List

CROP_METADATA: Dict[str, Dict[str, Any]] = {
    "garlic": {
        "en_name": "Garlic",
        "bn_name": "দেশি রসুন",
        "category_bn": "মশলা জাতীয় ফসল (Spices)",
        "category_en": "Spice Crop (Alliums)",
        "season_bn": "রবি মৌসুম (অক্টোবর - মার্চ)",
        "season_en": "Rabi Season (October - March)",
        "volatility_bn": "মাঝারি অস্থিরতা (Moderate Volatility)",
        "volatility_en": "Moderate Volatility",
        "key_districts_bn": "নাটোর, পাবনা, ফরিদপুর, মেহেরপুর",
        "key_districts_en": "Natore, Pabna, Faridpur, Meherpur",
        "storage_tip_bn": "পর্যাপ্ত বায়ুচলাচলযুক্ত শুষ্ক মাচা বা ঝুলন্ত জালে সংরক্ষণ করলে আর্দ্রতাজনিত পচন রোধ করা সম্ভব এবং ২-৩ মাস গুণমান অক্ষুণ্ণ থাকে।",
        "storage_tip_en": "Store in dry, well-ventilated lofts or suspended mesh nets to prevent fungal rot; preserves commercial grade for 2-3 months.",
        "unit": "BDT/kg",
        "unit_bn": "টাকা/কেজি",
        "icon": "🧄",
    },
    "chickpea": {
        "en_name": "Chickpea",
        "bn_name": "ছোলা / বুট",
        "category_bn": "ডাল জাতীয় ফসল (Pulses)",
        "category_en": "Pulse / Legume Crop",
        "season_bn": "রবি মৌসুম (নভেম্বর - এপ্রিল)",
        "season_en": "Rabi Season (November - April)",
        "volatility_bn": "স্থিতিশীল (Stable Commodity)",
        "volatility_en": "Stable Commodity",
        "key_districts_bn": "রাজশাহী, চাঁপাইনবাবগঞ্জ, যশোর, কুষ্টিয়া",
        "key_districts_en": "Rajshahi, Chapainawabganj, Jashore, Kushtia",
        "storage_tip_bn": "আর্দ্রতা ১০% এর নিচে নামিয়ে বায়ুরোধী সাইলো বা পাত্রে সংরক্ষণ করলে গুণমান ও পুষ্টিমান দীর্ঘদিন সুরক্ষিত থাকে।",
        "storage_tip_en": "Dry grain moisture below 10% and store in airtight silos or bins to prevent insect weevil infestation.",
        "unit": "BDT/kg",
        "unit_bn": "টাকা/কেজি",
        "icon": "🧆",
    },
    "green_chilli": {
        "en_name": "Green Chilli",
        "bn_name": "কাঁচামরিচ",
        "category_bn": "কাঁচা সবজি ও মশলা (Vegetable/Spice)",
        "category_en": "Fresh Spice / Vegetable",
        "season_bn": "খরিফ ও রবি (বছরব্যাপী চাষাবাদ)",
        "season_en": "Kharif & Rabi (Year-round Production)",
        "volatility_bn": "অত্যধিক উচ্চ অস্থিরতা (High Volatility)",
        "volatility_en": "Very High Volatility (Perishable)",
        "key_districts_bn": "বগুড়া, জামালপুর, পঞ্চগড়, চুয়াডাঙ্গা",
        "key_districts_en": "Bogura, Jamalpur, Panchagarh, Chuadanga",
        "storage_tip_bn": "উচ্চমাত্রায় পচনশীল বিধায় ফসল সংগ্রহের পর সরাসরি সূর্যালোক পরিহার করে গ্রেডিং শেষে দ্রুত নিকটস্থ পাইকারি মোকামে প্রেরণ আবশ্যক।",
        "storage_tip_en": "Highly perishable; avoid direct sunlight post-harvest, pre-cool, grade, and dispatch immediately to wholesale distribution centers.",
        "unit": "BDT/kg",
        "unit_bn": "টাকা/কেজি",
        "icon": "🌶️",
    },
    "cucumber": {
        "en_name": "Cucumber",
        "bn_name": "শসা ও খিরা",
        "category_bn": "ফলজাতীয় সবজি (Cucurbit Vegetable)",
        "category_en": "Salad / Cucurbit Vegetable",
        "season_bn": "খরিফ-১ ও রবি (মার্চ - সেপ্টেম্বর / শীতকালীন)",
        "season_en": "Kharif-1 & Rabi (March - September / Winter)",
        "volatility_bn": "উচ্চ অস্থিরতা (High Volatility)",
        "volatility_en": "High Volatility (Short Shelf-life)",
        "key_districts_bn": "কুমিল্লা, নরসিংদী, যশোর, বগুড়া",
        "key_districts_en": "Cumilla, Narsingdi, Jashore, Bogura",
        "storage_tip_bn": "সংরক্ষণ স্থায়িত্ব সীমিত বিধায় সংগ্রহের পর আর্দ্র-শীতল পরিবেশে রেখে স্বল্পতম সময়ে আড়ত ও বিপণন কেন্দ্রে সরবরাহ করা সমীচীন।",
        "storage_tip_en": "Limited shelf-life; maintain cool humid conditions and market within 24-48 hours to minimize transpirational weight loss.",
        "unit": "BDT/kg",
        "unit_bn": "টাকা/কেজি",
        "icon": "🥒",
    },
    "sweet_pumpkin": {
        "en_name": "Sweet Pumpkin",
        "bn_name": "মিষ্টি কুমড়া",
        "category_bn": "কন্দ ও লতাজাতীয় সবজি (Cucurbits)",
        "category_en": "Cucurbit / Storage Vegetable",
        "season_bn": "রবি ও খরিফ (নভেম্বর - জুন)",
        "season_en": "Rabi & Kharif (November - June)",
        "volatility_bn": "স্বল্প-মাঝারি অস্থিরতা (Low-Moderate Volatility)",
        "volatility_en": "Low-Moderate Volatility",
        "key_districts_bn": "কুড়িগ্রাম, গাইবান্ধা (চরাঞ্চল), রংপুর, ফরিদপুর",
        "key_districts_en": "Kurigram, Gaibandha (Char Areas), Rangpur, Faridpur",
        "storage_tip_bn": "বোঁটাসহ পরিপক্ব ফল সংগ্রহ করে শুষ্ক, ছায়াযুক্ত মেঝ বা বাঁশের মাচায় স্তূপাকারে রাখলে ৩-৪ মাস স্বাভাবিক গুণমান বজায় থাকে।",
        "storage_tip_en": "Harvest with intact peduncle (stem); cure and stack on elevated dry wooden platforms to store reliably for 3-4 months.",
        "unit": "BDT/kg",
        "unit_bn": "টাকা/কেজি",
        "icon": "🎃",
    },
}


def get_crop_info(crop: str) -> Dict[str, Any]:
    """Return agricultural metadata for a given crop identifier."""
    if crop not in CROP_METADATA:
        raise ValueError(f"Unknown crop: '{crop}'. Available crops: {list(CROP_METADATA.keys())}")
    info = dict(CROP_METADATA[crop])
    # Maintain backward compatibility for keys used in existing code/tests
    info["category"] = info["category_bn"]
    info["season_bn"] = info["season_bn"]
    info["volatility"] = info["volatility_bn"]
    info["key_districts"] = info["key_districts_bn"]
    info["storage_tip"] = info["storage_tip_bn"]
    return info


def get_crop_display_label(crop: str, lang: str = "bn") -> str:
    """Return formatted bilingual display label for dropdown selection."""
    info = get_crop_info(crop)
    icon = info.get("icon", "🌾")
    if lang == "en":
        return f"{icon} {info['en_name']} ({info['bn_name']})"
    return f"{icon} {info['bn_name']} ({info['en_name']})"


def list_available_crops() -> List[str]:
    """Return list of supported crop keys."""
    return list(CROP_METADATA.keys())
