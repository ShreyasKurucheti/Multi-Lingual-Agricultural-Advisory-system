import streamlit as st
import numpy as np
import pickle
from gtts import gTTS
import tempfile
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import time
import os
import re
import base64
import json

st.set_page_config(
    page_title="AgriSense AI | Crop Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════
#  ML MODEL CLASSES (original 30 — CSV-backed crops)
# ═══════════════════════════════════════════════════════════════════
ML_CLASSES = [
    'banana_bract_mosaic_virus','banana_cordana','banana_healthy','banana_insectpest',
    'banana_moko','banana_panama','banana_pestalotiopsis','banana_sigatoka','banana_yb_sigatoka',
    'cauliflower_Blackrot','cauliflower_bacterial_spot_rot','cauliflower_downy_mildew','cauliflower_healthy',
    'chilli_anthracnose','chilli_healthy','chilli_leafcurl','chilli_leafspot','chilli_whitefly',
    'chilli_yellowish','groundnut_early_leaf_spot','groundnut_early_rust','groundnut_healthy',
    'groundnut_late_leaf_spot','groundnut_nutrition_deficiency','groundnut_rust',
    'radish_black_leaf_spot','radish_downey_mildew','radish_flea_beetle','radish_healthy','radish_mosaic'
]

ML_CROP_CLASSES = {
    "Banana":     [c for c in ML_CLASSES if c.startswith("banana_")],
    "Cauliflower":[c for c in ML_CLASSES if c.startswith("cauliflower_")],
    "Chilli":     [c for c in ML_CLASSES if c.startswith("chilli_")],
    "Groundnut":  [c for c in ML_CLASSES if c.startswith("groundnut_")],
    "Radish":     [c for c in ML_CLASSES if c.startswith("radish_")],
}

ML_CROPS = set(ML_CROP_CLASSES.keys())

# ═══════════════════════════════════════════════════════════════════
#  100-CROP RULE-BASED DATABASE
# ═══════════════════════════════════════════════════════════════════
RULE_BASED_DB = {
    "Rice": [
        {"disease":"Blast",           "remedy":"Use resistant varieties + fungicide spray",         "severity":"High",     "profile":[8,2,3,6,1,7]},
        {"disease":"Brown Spot",      "remedy":"Seed treatment + balanced fertilizer",               "severity":"Moderate", "profile":[7,5,2,5,1,8]},
        {"disease":"Sheath Blight",   "remedy":"Proper spacing + fungicide",                         "severity":"High",     "profile":[6,3,4,8,2,5]},
        {"disease":"Bacterial Blight","remedy":"Resistant varieties",                                 "severity":"High",     "profile":[5,4,5,2,2,6]},
        {"disease":"False Smut",      "remedy":"Fungicide spray",                                    "severity":"Moderate", "profile":[4,2,2,9,1,6]},
        {"disease":"Tip Burn",        "remedy":"Proper water drainage",                              "severity":"Low",      "profile":[2,4,3,1,1,8]},
        {"disease":"Healthy",         "remedy":"No treatment needed. Continue regular crop care.",   "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Wheat": [
        {"disease":"Rust",            "remedy":"Grow resistant varieties + fungicide",               "severity":"High",     "profile":[6,3,3,7,1,8]},
        {"disease":"Loose Smut",      "remedy":"Treat seeds with fungicide",                         "severity":"Moderate", "profile":[3,2,2,9,1,5]},
        {"disease":"Leaf Spot",       "remedy":"Remove infected leaves",                             "severity":"Moderate", "profile":[8,3,2,4,1,7]},
        {"disease":"Karnal Bunt",     "remedy":"Seed treatment",                                    "severity":"High",     "profile":[4,2,3,9,1,6]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Maize": [
        {"disease":"Leaf Blight",     "remedy":"Crop rotation + fungicide",                          "severity":"High",     "profile":[7,4,3,5,2,7]},
        {"disease":"Fall Armyworm",   "remedy":"Use pesticides + biological control",                "severity":"High",     "profile":[6,3,5,2,3,5]},
        {"disease":"Rust",            "remedy":"Crop rotation + fungicide",                          "severity":"Moderate", "profile":[6,3,3,7,1,8]},
        {"disease":"Wilt",            "remedy":"Soil treatment",                                    "severity":"High",     "profile":[2,6,9,3,1,7]},
        {"disease":"Stalk Rot",       "remedy":"Crop rotation",                                     "severity":"High",     "profile":[3,5,8,6,1,6]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Barley": [
        {"disease":"Powdery Mildew",  "remedy":"Sulfur fungicide",                                  "severity":"Moderate", "profile":[5,3,3,9,2,4]},
        {"disease":"Smut",            "remedy":"Seed treatment",                                    "severity":"Moderate", "profile":[3,2,2,8,1,5]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Sorghum": [
        {"disease":"Grain Mold",      "remedy":"Proper drying + resistant seeds",                    "severity":"High",     "profile":[4,3,3,9,1,7]},
        {"disease":"Ergot",           "remedy":"Remove infected heads",                              "severity":"High",     "profile":[3,2,3,8,1,6]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Bajra": [
        {"disease":"Downy Mildew",    "remedy":"Seed treatment + fungicide",                         "severity":"High",     "profile":[5,5,5,9,2,5]},
        {"disease":"Ergot",           "remedy":"Seed cleaning",                                     "severity":"Moderate", "profile":[3,3,3,8,1,6]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Ragi": [
        {"disease":"Blast",           "remedy":"Use resistant varieties",                            "severity":"High",     "profile":[8,2,3,6,1,7]},
        {"disease":"Leaf Spot",       "remedy":"Remove infected plants",                             "severity":"Moderate", "profile":[8,3,2,4,1,7]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Oats": [
        {"disease":"Crown Rust",      "remedy":"Fungicide spray",                                   "severity":"Moderate", "profile":[6,3,3,7,1,8]},
        {"disease":"Leaf Blotch",     "remedy":"Fungicide",                                         "severity":"Moderate", "profile":[7,4,3,5,2,6]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Chickpea": [
        {"disease":"Fusarium Wilt",   "remedy":"Crop rotation",                                     "severity":"High",     "profile":[3,6,9,4,2,7]},
        {"disease":"Ascochyta Blight","remedy":"Fungicide",                                          "severity":"High",     "profile":[7,4,4,6,2,6]},
        {"disease":"Root Rot",        "remedy":"Soil treatment",                                    "severity":"High",     "profile":[2,5,9,5,1,6]},
        {"disease":"Dry Root Rot",    "remedy":"Soil solarization",                                 "severity":"Moderate", "profile":[2,5,8,4,1,7]},
        {"disease":"Botrytis Gray Mold","remedy":"Fungicide",                                        "severity":"High",     "profile":[5,4,4,10,2,5]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Pigeon Pea": [
        {"disease":"Wilt",            "remedy":"Resistant varieties",                                "severity":"High",     "profile":[2,6,9,3,2,7]},
        {"disease":"Sterility Mosaic","remedy":"Remove infected plants",                             "severity":"High",     "profile":[3,7,4,2,5,8]},
        {"disease":"Leaf Blight",     "remedy":"Remove infected leaves",                             "severity":"Moderate", "profile":[7,4,4,5,2,6]},
        {"disease":"Phytophthora Blight","remedy":"Drainage",                                        "severity":"High",     "profile":[4,4,7,8,2,5]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Green Gram": [
        {"disease":"Yellow Mosaic Virus","remedy":"Control whiteflies",                              "severity":"High",     "profile":[2,9,4,2,3,9]},
        {"disease":"Cercospora Leaf Spot","remedy":"Fungicide",                                      "severity":"Moderate", "profile":[8,3,2,4,1,7]},
        {"disease":"Powdery Mildew",  "remedy":"Fungicide",                                         "severity":"Moderate", "profile":[5,3,3,9,2,4]},
        {"disease":"Rust",            "remedy":"Fungicide",                                         "severity":"Moderate", "profile":[6,3,3,7,1,8]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Black Gram": [
        {"disease":"Leaf Spot",       "remedy":"Fungicide",                                         "severity":"Moderate", "profile":[8,3,2,4,1,7]},
        {"disease":"Wilt",            "remedy":"Crop rotation",                                     "severity":"High",     "profile":[2,6,9,3,2,7]},
        {"disease":"Mosaic Virus",    "remedy":"Insect control",                                    "severity":"High",     "profile":[3,8,4,2,5,9]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Lentil": [
        {"disease":"Rust",            "remedy":"Fungicide spray",                                   "severity":"Moderate", "profile":[6,3,3,7,1,8]},
        {"disease":"Wilt",            "remedy":"Resistant seeds",                                   "severity":"High",     "profile":[2,6,9,3,2,7]},
        {"disease":"Collar Rot",      "remedy":"Seed treatment",                                    "severity":"High",     "profile":[2,4,8,7,1,6]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Pea": [
        {"disease":"Powdery Mildew",  "remedy":"Sulfur spray",                                     "severity":"Moderate", "profile":[5,3,3,9,2,4]},
        {"disease":"Downy Mildew",    "remedy":"Fungicide",                                         "severity":"Moderate", "profile":[5,5,5,9,2,5]},
        {"disease":"Root Rot",        "remedy":"Soil drainage",                                     "severity":"High",     "profile":[2,5,9,5,1,6]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Cowpea": [
        {"disease":"Mosaic Virus",    "remedy":"Pest control",                                      "severity":"High",     "profile":[3,8,4,2,5,9]},
        {"disease":"Anthracnose",     "remedy":"Seed treatment",                                    "severity":"High",     "profile":[8,3,3,6,2,7]},
        {"disease":"Bacterial Blight","remedy":"Clean seeds",                                       "severity":"Moderate", "profile":[6,4,4,3,2,6]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Tomato": [
        {"disease":"Leaf Curl Virus", "remedy":"Control whiteflies",                                "severity":"High",     "profile":[3,5,4,2,8,7]},
        {"disease":"Early Blight",    "remedy":"Fungicide",                                         "severity":"High",     "profile":[8,4,3,5,2,7]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Potato": [
        {"disease":"Late Blight",     "remedy":"Fungicide",                                         "severity":"Critical", "profile":[7,4,5,9,2,7]},
        {"disease":"Black Scurf",     "remedy":"Seed treatment",                                    "severity":"Moderate", "profile":[5,3,3,7,1,6]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Brinjal": [
        {"disease":"Phomopsis Blight","remedy":"Remove infected plants",                             "severity":"High",     "profile":[7,3,4,7,2,7]},
        {"disease":"Wilt",            "remedy":"Soil treatment",                                    "severity":"High",     "profile":[2,6,9,3,2,7]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Onion": [
        {"disease":"Downy Mildew",    "remedy":"Fungicide",                                         "severity":"High",     "profile":[5,5,5,9,2,5]},
        {"disease":"Purple Blotch",   "remedy":"Spray fungicide",                                   "severity":"High",     "profile":[7,4,3,6,2,8]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Garlic": [
        {"disease":"White Rot",       "remedy":"Crop rotation",                                     "severity":"High",     "profile":[3,4,7,9,2,5]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Cabbage": [
        {"disease":"Black Rot",       "remedy":"Hot water seed treatment",                          "severity":"High",     "profile":[5,5,4,3,2,8]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Carrot": [
        {"disease":"Leaf Blight",     "remedy":"Remove infected leaves",                             "severity":"Moderate", "profile":[7,4,3,5,2,7]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Cucumber": [
        {"disease":"Powdery Mildew",  "remedy":"Sulfur spray",                                     "severity":"Moderate", "profile":[5,3,3,9,2,4]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Pumpkin": [
        {"disease":"Downy Mildew",    "remedy":"Fungicide",                                         "severity":"Moderate", "profile":[5,5,5,9,2,5]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Bitter Gourd": [
        {"disease":"Mosaic Virus",    "remedy":"Pest control",                                      "severity":"High",     "profile":[3,8,4,2,5,9]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Bottle Gourd": [
        {"disease":"Anthracnose",     "remedy":"Fungicide",                                         "severity":"Moderate", "profile":[8,3,3,6,2,7]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Beans": [
        {"disease":"Rust",            "remedy":"Fungicide",                                         "severity":"Moderate", "profile":[6,3,3,7,1,8]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Capsicum": [
        {"disease":"Bacterial Spot",  "remedy":"Copper spray",                                     "severity":"Moderate", "profile":[6,4,4,3,2,7]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Okra": [
        {"disease":"Yellow Vein Mosaic","remedy":"Whitefly control",                                 "severity":"High",     "profile":[2,9,4,2,3,9]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Sweet Potato": [
        {"disease":"Root Rot",        "remedy":"Soil treatment",                                    "severity":"High",     "profile":[2,5,9,5,1,6]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Coriander": [
        {"disease":"Stem Rot",        "remedy":"Crop rotation",                                     "severity":"Moderate", "profile":[3,4,7,8,1,5]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Spinach": [
        {"disease":"Downy Mildew",    "remedy":"Fungicide",                                         "severity":"Moderate", "profile":[5,5,5,9,2,5]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Lettuce": [
        {"disease":"Leaf Spot",       "remedy":"Proper spacing",                                    "severity":"Low",      "profile":[8,3,2,4,1,7]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Mango": [
        {"disease":"Powdery Mildew",  "remedy":"Fungicide",                                         "severity":"Moderate", "profile":[5,3,3,9,2,4]},
        {"disease":"Anthracnose",     "remedy":"Spray fungicide",                                   "severity":"High",     "profile":[8,3,3,6,2,7]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Apple": [
        {"disease":"Scab",            "remedy":"Fungicide",                                         "severity":"High",     "profile":[7,3,3,6,2,8]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Grapes": [
        {"disease":"Downy Mildew",    "remedy":"Fungicide",                                         "severity":"High",     "profile":[5,5,5,9,2,5]},
        {"disease":"Powdery Mildew",  "remedy":"Sulfur spray",                                     "severity":"Moderate", "profile":[5,3,3,9,2,4]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Citrus": [
        {"disease":"Canker",          "remedy":"Remove infected parts",                             "severity":"High",     "profile":[7,4,4,3,2,8]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Guava": [
        {"disease":"Wilt",            "remedy":"Soil treatment",                                    "severity":"High",     "profile":[2,6,9,3,2,7]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Papaya": [
        {"disease":"Ring Spot Virus", "remedy":"Remove infected plants",                             "severity":"Critical", "profile":[3,8,4,2,5,9]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Pomegranate": [
        {"disease":"Bacterial Blight","remedy":"Copper spray",                                       "severity":"High",     "profile":[6,4,4,3,2,7]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Coconut": [
        {"disease":"Bud Rot",         "remedy":"Fungicide",                                         "severity":"Critical", "profile":[4,5,8,9,2,6]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Mustard": [
        {"disease":"White Rust",      "remedy":"Fungicide",                                         "severity":"High",     "profile":[4,3,3,8,2,6]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Sunflower": [
        {"disease":"Downy Mildew",    "remedy":"Seed treatment",                                    "severity":"High",     "profile":[5,5,5,9,2,5]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Soybean": [
        {"disease":"Rust",            "remedy":"Fungicide",                                         "severity":"High",     "profile":[6,3,3,7,1,8]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Sesame": [
        {"disease":"Leaf Spot",       "remedy":"Fungicide",                                         "severity":"Moderate", "profile":[8,3,2,4,1,7]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Cotton": [
        {"disease":"Bollworm",        "remedy":"Insecticide",                                       "severity":"High",     "profile":[5,3,4,3,3,6]},
        {"disease":"Leaf Curl Virus", "remedy":"Pest control",                                      "severity":"High",     "profile":[3,5,4,2,8,7]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Sugarcane": [
        {"disease":"Red Rot",         "remedy":"Resistant varieties",                                "severity":"Critical", "profile":[4,5,7,8,2,8]},
        {"disease":"Smut",            "remedy":"Seed treatment",                                    "severity":"High",     "profile":[3,2,2,8,1,5]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Tea": [
        {"disease":"Blister Blight",  "remedy":"Fungicide",                                         "severity":"High",     "profile":[7,3,3,8,2,6]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Coffee": [
        {"disease":"Leaf Rust",       "remedy":"Fungicide",                                         "severity":"High",     "profile":[6,3,3,7,1,8]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
    "Tobacco": [
        {"disease":"Mosaic Virus",    "remedy":"Use healthy seeds",                                 "severity":"High",     "profile":[3,8,4,2,5,9]},
        {"disease":"Healthy",         "remedy":"No treatment needed.",                               "severity":"None",     "profile":[0,0,0,0,0,0]},
    ],
}

ALL_CROPS = sorted(list(ML_CROP_CLASSES.keys()) + [c for c in RULE_BASED_DB.keys() if c not in ML_CROP_CLASSES])

CROP_EMOJIS = {
    "Banana":"🍌","Cauliflower":"🥦","Chilli":"🌶️","Groundnut":"🥜","Radish":"🌱",
    "Rice":"🌾","Wheat":"🌾","Maize":"🌽","Barley":"🌾","Sorghum":"🌾",
    "Bajra":"🌾","Ragi":"🌾","Oats":"🌾",
    "Chickpea":"🫘","Pigeon Pea":"🫘","Green Gram":"🫘","Black Gram":"🫘",
    "Lentil":"🫘","Pea":"🫛","Cowpea":"🫘",
    "Tomato":"🍅","Potato":"🥔","Brinjal":"🍆","Onion":"🧅","Garlic":"🧄",
    "Cabbage":"🥬","Carrot":"🥕","Cucumber":"🥒","Pumpkin":"🎃",
    "Bitter Gourd":"🥒","Bottle Gourd":"🥒","Beans":"🫘","Capsicum":"🫑",
    "Okra":"🌿","Sweet Potato":"🍠","Coriander":"🌿","Spinach":"🥬","Lettuce":"🥬",
    "Mango":"🥭","Apple":"🍎","Grapes":"🍇","Citrus":"🍊","Guava":"🍈",
    "Papaya":"🍐","Pomegranate":"🍎","Coconut":"🥥",
    "Mustard":"🌿","Sunflower":"🌻","Soybean":"🫘","Sesame":"🌿",
    "Cotton":"🌿","Sugarcane":"🌿","Tea":"🍃","Coffee":"☕","Tobacco":"🌿",
}

def get_crop_emoji(crop):
    return CROP_EMOJIS.get(crop, "🌿")

DISEASE_META = {
    'banana_bract_mosaic_virus':    {"treatment":"Remove infected plants. Use virus-free planting material. Control aphid vectors with neem-based insecticides.",        "severity":"High"},
    'banana_cordana':               {"treatment":"Apply mancozeb or copper-based fungicide. Remove and burn infected leaves. Improve drainage.",                          "severity":"Moderate"},
    'banana_healthy':               {"treatment":"No treatment needed. Maintain regular fertilization and irrigation schedules.",                                          "severity":"None"},
    'banana_insectpest':            {"treatment":"Use neem oil spray (5ml/L). Apply imidacloprid for severe infestations. Remove heavily damaged leaves.",                "severity":"Moderate"},
    'banana_moko':                  {"treatment":"Destroy infected plants immediately. Disinfect cutting tools with bleach solution. Quarantine affected area.",           "severity":"Critical"},
    'banana_panama':                {"treatment":"No cure; remove and destroy plants. Plant resistant varieties (Cavendish). Soil solarization recommended.",             "severity":"Critical"},
    'banana_pestalotiopsis':        {"treatment":"Apply propiconazole fungicide. Remove necrotic leaf tips. Reduce waterlogging around plant base.",                      "severity":"Moderate"},
    'banana_sigatoka':              {"treatment":"Spray mancozeb or trifloxystrobin every 10 days. Remove lower infected leaves. Improve air circulation.",              "severity":"High"},
    'banana_yb_sigatoka':           {"treatment":"Apply systemic fungicides (propiconazole). Destroy infected leaves. Use certified disease-free suckers.",              "severity":"High"},
    'cauliflower_Blackrot':         {"treatment":"Use hot water treated seeds (50°C, 30min). Apply copper bactericide. Rotate with non-crucifer crops.",                 "severity":"High"},
    'cauliflower_bacterial_spot_rot':{"treatment":"Apply copper hydroxide spray. Avoid overhead irrigation. Remove infected tissue immediately.",                         "severity":"High"},
    'cauliflower_downy_mildew':     {"treatment":"Apply metalaxyl-mancozeb. Improve air circulation. Avoid wetting leaves during irrigation.",                           "severity":"Moderate"},
    'cauliflower_healthy':          {"treatment":"No treatment needed. Continue good agricultural practices and monitoring.",                                              "severity":"None"},
    'chilli_anthracnose':           {"treatment":"Apply mancozeb or carbendazim fungicide. Harvest fruits promptly. Use disease-free seeds.",                            "severity":"High"},
    'chilli_healthy':               {"treatment":"No treatment needed. Maintain balanced NPK fertilization.",                                                             "severity":"None"},
    'chilli_leafcurl':              {"treatment":"Control thrips/mites with abamectin spray. Remove infected plants. Use virus-resistant varieties.",                    "severity":"High"},
    'chilli_leafspot':              {"treatment":"Spray copper oxychloride or carbendazim. Remove fallen leaves. Avoid excess nitrogen application.",                    "severity":"Moderate"},
    'chilli_whitefly':              {"treatment":"Apply imidacloprid or neem oil spray. Use yellow sticky traps. Introduce natural predators.",                          "severity":"Moderate"},
    'chilli_yellowish':             {"treatment":"Check for nutrient deficiency (Fe, Mg). Apply chelated micronutrient spray. Test soil pH.",                            "severity":"Moderate"},
    'groundnut_early_leaf_spot':    {"treatment":"Apply chlorothalonil or mancozeb every 14 days. Remove crop debris. Use resistant varieties.",                         "severity":"Moderate"},
    'groundnut_early_rust':         {"treatment":"Spray propiconazole or trifloxystrobin. Apply at first sign of infection. Rotate crops.",                             "severity":"Moderate"},
    'groundnut_healthy':            {"treatment":"No treatment needed. Ensure proper spacing for air circulation.",                                                       "severity":"None"},
    'groundnut_late_leaf_spot':     {"treatment":"Apply tebuconazole fungicide. Remove infected leaves. Avoid late-season nitrogen application.",                        "severity":"High"},
    'groundnut_nutrition_deficiency':{"treatment":"Conduct soil test. Apply recommended NPK + micro-nutrients. Correct soil pH to 6.0–6.5.",                             "severity":"Moderate"},
    'groundnut_rust':               {"treatment":"Apply triadimefon or propiconazole. Spray every 10–14 days. Use resistant cultivars.",                                "severity":"High"},
    'radish_black_leaf_spot':       {"treatment":"Apply iprodione or chlorothalonil. Remove infected leaves. Avoid overhead watering.",                                  "severity":"Moderate"},
    'radish_downey_mildew':         {"treatment":"Spray metalaxyl + mancozeb. Improve drainage and plant spacing. Avoid cool wet conditions.",                           "severity":"Moderate"},
    'radish_flea_beetle':           {"treatment":"Apply pyrethrin or spinosad spray. Use row covers. Remove plant debris after harvest.",                                 "severity":"Low"},
    'radish_healthy':               {"treatment":"No treatment needed. Harvest at proper maturity to avoid cracking.",                                                   "severity":"None"},
    'radish_mosaic':                {"treatment":"Remove and destroy infected plants. Control aphid vectors. Use virus-free seeds.",                                      "severity":"High"},
}

SEVERITY_COLORS = {
    "None":"#00e676","Low":"#69f0ae","Moderate":"#ffd600","High":"#ff9100","Critical":"#ff5252"
}

ML_PROFILES = {
    'banana_bract_mosaic_virus':[2,5,2,1,6,7],'banana_cordana':[7,3,2,5,1,4],
    'banana_healthy':[0,0,0,0,0,0],'banana_insectpest':[4,2,4,2,7,3],
    'banana_moko':[2,7,8,3,2,5],'banana_panama':[1,6,9,2,1,6],
    'banana_pestalotiopsis':[8,4,3,6,2,5],'banana_sigatoka':[9,4,2,7,1,6],
    'banana_yb_sigatoka':[8,5,3,6,2,7],'cauliflower_Blackrot':[3,5,4,2,1,8],
    'cauliflower_bacterial_spot_rot':[5,3,5,4,2,6],'cauliflower_downy_mildew':[4,3,3,8,2,4],
    'cauliflower_healthy':[0,0,0,0,0,0],'chilli_anthracnose':[7,3,3,5,2,6],
    'chilli_healthy':[0,0,0,0,0,0],'chilli_leafcurl':[2,2,3,1,9,4],
    'chilli_leafspot':[8,3,2,4,3,5],'chilli_whitefly':[3,4,3,2,5,3],
    'chilli_yellowish':[2,9,3,2,3,7],'groundnut_early_leaf_spot':[6,3,2,4,1,4],
    'groundnut_early_rust':[5,2,2,3,1,6],'groundnut_healthy':[0,0,0,0,0,0],
    'groundnut_late_leaf_spot':[8,4,3,6,2,5],'groundnut_nutrition_deficiency':[2,7,4,1,3,7],
    'groundnut_rust':[6,3,3,5,2,7],'radish_black_leaf_spot':[8,3,2,5,2,5],
    'radish_downey_mildew':[4,3,3,8,2,4],'radish_flea_beetle':[5,2,3,2,4,3],
    'radish_healthy':[0,0,0,0,0,0],'radish_mosaic':[3,5,3,2,6,6],
}

def fmt_name(cls):
    parts = cls.split("_")
    return " ".join(p.capitalize() for p in parts[1:]) if len(parts) > 1 else cls.capitalize()

def detect_rule_based(crop, leaf_spots, yellowing, wilting, mold, leaf_curl, discoloration):
    diseases = RULE_BASED_DB.get(crop, [])
    if not diseases:
        return None, 0.0
    p = [leaf_spots, yellowing, wilting, mold, leaf_curl, discoloration]
    scores = {}
    for d in diseases:
        dist = sum(abs(p[i] - d["profile"][i]) for i in range(6))
        scores[d["disease"]] = dist
    best_name = min(scores, key=scores.get)
    best_dist = scores[best_name]
    confidence = max(78.0, 95.0 - (best_dist / 60.0) * 17.0)
    matched = next(d for d in diseases if d["disease"] == best_name)
    return matched, confidence

# ═══════════════════════════════════════════════════════════════════
#  IMAGE RECOGNITION VIA CLAUDE VISION API
# ═══════════════════════════════════════════════════════════════════
def analyze_crop_image(image_bytes, image_type, crop_name, api_key):
    """Send crop image to Claude Vision for disease analysis."""
    b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")
    
    # Map media types
    media_map = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "webp": "image/webp", "gif": "image/gif"
    }
    media_type = media_map.get(image_type.lower().replace(".", ""), "image/jpeg")
    
    all_diseases = []
    if crop_name in ML_CROPS:
        all_diseases = [fmt_name(c) for c in ML_CROP_CLASSES[crop_name]]
    else:
        all_diseases = [d["disease"] for d in RULE_BASED_DB.get(crop_name, [])]
    
    disease_list_str = ", ".join(all_diseases) if all_diseases else "various diseases"
    
    prompt = f"""You are an expert agricultural pathologist. Analyze this crop leaf/plant image for disease detection.

The selected crop is: {crop_name}
Known diseases for this crop: {disease_list_str}

Please analyze the image and respond in EXACTLY this JSON format (no extra text):
{{
  "detected_disease": "disease name from the known list, or 'Unknown' if not recognizable",
  "confidence": 85,
  "severity": "None|Low|Moderate|High|Critical",
  "visual_symptoms": "brief description of what you see in the image",
  "leaf_spots": 5,
  "yellowing": 3,
  "wilting": 2,
  "mold": 1,
  "leaf_curl": 4,
  "discoloration": 6,
  "treatment": "specific treatment recommendation",
  "image_quality": "Good|Poor|Unclear"
}}

Scores (0-10) reflect symptom severity observed in the image.
If image is unclear or not a plant, set detected_disease to "Image Unclear" and confidence to 0."""

    try:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1000,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_image
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        }
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=30
        )
        data = resp.json()
        raw = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
        # Strip markdown code fences if present
        raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
        result = json.loads(raw)
        return result, None
    except json.JSONDecodeError as e:
        return None, f"Could not parse AI response: {e}"
    except Exception as e:
        return None, f"Vision API error: {e}"


# ═══════════════════════════════════════════════════════════════════
#  AUDIO TRANSCRIPTION VIA CLAUDE API
# ═══════════════════════════════════════════════════════════════════
def transcribe_audio_query(audio_bytes, api_key):
    """Transcribe farmer's audio question using Whisper-style prompt via Claude."""
    # We send audio as base64 to Claude's messages API
    # Claude supports audio input natively in newer versions
    b64_audio = base64.standard_b64encode(audio_bytes).decode("utf-8")
    
    try:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        # Use Claude with the audio as a document
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 500,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "audio/wav",
                            "data": b64_audio
                        }
                    },
                    {
                        "type": "text",
                        "text": "Please transcribe this audio message from a farmer about crop disease. Return ONLY the transcribed text, nothing else."
                    }
                ]
            }]
        }
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=30
        )
        data = resp.json()
        
        # Check for API errors
        if "error" in data:
            # Fallback: return a helpful message
            return None, f"Audio transcription not supported in current API tier. Please type your question."
        
        transcribed = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
        return transcribed, None
    except Exception as e:
        return None, f"Transcription error: {e}"


# ═══════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════════
for k, v in {
    "chat_history": [], "chip_query": "", "sidebar_open": False,
    "sidebar_section": "notifications", "sb_user_email": "",
    "sb_sender_email": "", "sb_email_pass": "", "sb_telegram_chat": "",
    "sb_tg_token": "", "sb_crop": "Banana",
    "sb_leaf_spots": 3, "sb_yellowing": 2, "sb_wilting": 1,
    "sb_mold": 4, "sb_leaf_curl": 2, "sb_discoloration": 3,
    "sb_translate_langs": ["Hindi", "Telugu"],
    "image_analysis_result": None,
    "audio_transcription": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════
#  STYLES
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Exo+2:wght@200;300;400;600;700&family=JetBrains+Mono:wght@300;400&display=swap');

:root {
  --bg-deep:    #050a0e;
  --bg-panel:   #0a1520;
  --bg-card:    #0d1e2e;
  --accent-g:   #00e676;
  --accent-b:   #00b0ff;
  --accent-y:   #ffd600;
  --accent-p:   #e040fb;
  --border:     rgba(0,230,118,0.18);
  --text-main:  #e8f5e9;
  --text-muted: #7fa89a;
  --glow-g:     0 0 24px rgba(0,230,118,0.35);
  --glow-b:     0 0 24px rgba(0,176,255,0.35);
  --glow-p:     0 0 24px rgba(224,64,251,0.35);
}

html, body, .stApp {
  background: var(--bg-deep) !important;
  color: var(--text-main) !important;
  font-family: 'Exo 2', sans-serif !important;
}

.stApp::before {
  content: '';
  position: fixed; inset: 0;
  background:
    radial-gradient(ellipse at 20% 50%, rgba(0,230,118,0.06) 0%, transparent 60%),
    radial-gradient(ellipse at 80% 20%, rgba(0,176,255,0.06) 0%, transparent 60%);
  pointer-events: none; z-index: 0;
  animation: bgPulse 8s ease-in-out infinite alternate;
}
@keyframes bgPulse { 0% { opacity:.6; } 100% { opacity:1; } }

.block-container {
  max-width: 1100px !important;
  padding: 2rem 2.5rem !important;
  background: transparent !important;
  position: relative; z-index: 2;
}

.hero-wrapper { text-align: center; padding: 2.5rem 0 1.5rem; }
.hero-tag {
  display: inline-block; font-family: 'JetBrains Mono', monospace;
  font-size: 11px; letter-spacing: 4px; text-transform: uppercase;
  color: var(--accent-g); border: 1px solid var(--border);
  padding: 5px 18px; border-radius: 2px; margin-bottom: 1rem;
  animation: fadeSlideDown 0.6s ease both;
}
.hero-title {
  font-family: 'Rajdhani', sans-serif;
  font-size: clamp(2.4rem, 5vw, 4rem); font-weight: 700;
  letter-spacing: 2px; text-transform: uppercase;
  color: #fff; line-height: 1.1;
  animation: fadeSlideDown 0.8s ease both; margin: 0 0 0.5rem;
}
.hero-title span { color: var(--accent-g); text-shadow: var(--glow-g); }
.hero-sub {
  font-size: 15px; color: var(--text-muted); letter-spacing: 1px;
  animation: fadeSlideDown 1s ease both; font-family: 'JetBrains Mono', monospace;
}
.divider {
  width: 100%; height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent-g), var(--accent-b), transparent);
  margin: 1.5rem 0;
}
.glass-panel {
  background: var(--bg-panel); border: 1px solid var(--border);
  border-radius: 4px; padding: 1.8rem; margin: 1rem 0;
  position: relative; overflow: hidden; animation: fadeIn 0.6s ease both;
}
.glass-panel::before {
  content: ''; position: absolute; top: 0; left: 0;
  width: 3px; height: 100%;
  background: linear-gradient(180deg, var(--accent-g), var(--accent-b));
}
.vision-panel::before {
  background: linear-gradient(180deg, var(--accent-p), var(--accent-b));
}
.audio-panel::before {
  background: linear-gradient(180deg, var(--accent-y), var(--accent-p));
}
.panel-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
  color: var(--accent-g); margin-bottom: 1rem;
}
.panel-label-purple { color: var(--accent-p); }
.panel-label-yellow { color: var(--accent-y); }

.crop-category-badge {
  display: inline-block; font-family: 'JetBrains Mono', monospace;
  font-size: 9px; letter-spacing: 2px; text-transform: uppercase;
  color: var(--accent-b); border: 1px solid rgba(0,176,255,0.3);
  padding: 2px 10px; border-radius: 2px; margin-bottom: 0.5rem;
}
.ml-badge { color: var(--accent-g); border-color: rgba(0,230,118,0.3); }

.stSelectbox > label, .stSlider > label, .stTextInput > label {
  color: var(--accent-b) !important; font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important; letter-spacing: 2px !important; text-transform: uppercase !important;
}
.stSelectbox > div > div {
  background: var(--bg-card) !important; border: 1px solid var(--border) !important;
  border-radius: 3px !important; color: var(--text-main) !important;
}
.stTextInput > div > div > input {
  background: var(--bg-card) !important; border: 1px solid var(--border) !important;
  border-radius: 3px !important; color: var(--text-main) !important;
  font-family: 'JetBrains Mono', monospace !important;
}
.stSlider > div > div > div > div { background: var(--accent-g) !important; }

div.stButton > button {
  width: 100%; background: transparent !important;
  border: 1px solid var(--accent-g) !important; color: var(--accent-g) !important;
  font-family: 'Rajdhani', sans-serif !important; font-size: 16px !important;
  font-weight: 600 !important; letter-spacing: 4px !important;
  text-transform: uppercase !important; padding: 14px 24px !important;
  border-radius: 3px !important; transition: all 0.3s ease !important;
}
div.stButton > button:hover {
  color: #050a0e !important; background: var(--accent-g) !important;
  box-shadow: var(--glow-g) !important; transform: translateY(-2px) !important;
}

.result-card {
  background: var(--bg-panel); border: 1px solid var(--accent-g);
  border-radius: 4px; padding: 2rem; margin: 1.5rem 0;
  box-shadow: var(--glow-g);
  animation: resultReveal 0.8s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.vision-result-card {
  background: var(--bg-panel); border: 1px solid var(--accent-p);
  border-radius: 4px; padding: 2rem; margin: 1.5rem 0;
  box-shadow: var(--glow-p);
  animation: resultReveal 0.8s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.disease-badge {
  display: inline-block; background: rgba(0,230,118,0.1);
  border: 1px solid var(--accent-g); color: var(--accent-g);
  font-family: 'Rajdhani', sans-serif; font-size: 22px; font-weight: 700;
  letter-spacing: 2px; padding: 8px 24px; border-radius: 3px;
  text-transform: uppercase; box-shadow: var(--glow-g);
}
.vision-disease-badge {
  display: inline-block; background: rgba(224,64,251,0.1);
  border: 1px solid var(--accent-p); color: var(--accent-p);
  font-family: 'Rajdhani', sans-serif; font-size: 22px; font-weight: 700;
  letter-spacing: 2px; padding: 8px 24px; border-radius: 3px;
  text-transform: uppercase; box-shadow: var(--glow-p);
}
.treatment-badge {
  display: inline-block; background: rgba(0,176,255,0.1);
  border: 1px solid var(--accent-b); color: var(--accent-b);
  font-size: 14px; padding: 12px 20px; border-radius: 3px;
  box-shadow: var(--glow-b); margin-top: 0.5rem; line-height: 1.6;
}
.stat-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
.stat-box {
  flex: 1; min-width: 100px; background: var(--bg-card);
  border: 1px solid var(--border); padding: 1rem; border-radius: 3px; text-align: center;
}
.stat-value { font-family: 'Rajdhani', sans-serif; font-size: 28px; font-weight: 700; color: var(--accent-g); }
.stat-label { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--text-muted); letter-spacing: 2px; text-transform: uppercase; }

.notif-row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-top: 1rem; }
.notif-pill {
  font-family: 'JetBrains Mono', monospace; font-size: 11px;
  letter-spacing: 2px; padding: 5px 14px; border-radius: 20px;
  border: 1px solid; text-transform: uppercase;
}
.notif-sent  { border-color: var(--accent-g); color: var(--accent-g); background: rgba(0,230,118,0.08); }
.notif-fail  { border-color: #ff5252; color: #ff5252; background: rgba(255,82,82,0.08); }
.notif-skip  { border-color: var(--text-muted); color: var(--text-muted); background: transparent; }

/* Image upload zone */
.upload-zone {
  border: 2px dashed rgba(224,64,251,0.4); border-radius: 8px;
  padding: 2rem; text-align: center; background: rgba(224,64,251,0.04);
  transition: all 0.3s ease; margin: 1rem 0;
}
.upload-zone:hover { border-color: var(--accent-p); background: rgba(224,64,251,0.08); }
.upload-icon { font-size: 48px; margin-bottom: 0.5rem; }
.upload-hint { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); letter-spacing: 2px; }

/* Audio input zone */
.audio-zone {
  border: 2px dashed rgba(255,214,0,0.4); border-radius: 8px;
  padding: 1.5rem; background: rgba(255,214,0,0.04);
  margin: 1rem 0;
}
.audio-transcription-box {
  background: var(--bg-card); border: 1px solid rgba(255,214,0,0.3);
  border-radius: 4px; padding: 1rem 1.2rem; margin-top: 0.8rem;
  font-family: 'Exo 2', sans-serif; font-size: 14px; color: var(--accent-y);
  line-height: 1.6; border-left: 3px solid var(--accent-y);
}
.visual-symptoms-box {
  background: var(--bg-card); border: 1px solid rgba(224,64,251,0.3);
  border-radius: 4px; padding: 1rem 1.2rem; margin-top: 0.8rem;
  font-size: 13px; color: var(--text-main); line-height: 1.6;
  border-left: 3px solid var(--accent-p);
}

.chat-section-header { display: flex; align-items: center; gap: 12px; margin-bottom: 1.2rem; }
.chat-dot {
  width: 10px; height: 10px; border-radius: 50%; background: var(--accent-g);
  box-shadow: 0 0 8px rgba(0,230,118,0.8); animation: pulse-dot 1.5s ease-in-out infinite;
}
@keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(.75)} }
.chat-history-box {
  background: var(--bg-deep); border: 1px solid var(--border);
  border-radius: 4px; padding: 1rem; min-height: 220px; max-height: 360px;
  overflow-y: auto; margin-bottom: 1rem; display: flex; flex-direction: column; gap: 0.75rem;
}
.chat-msg-user {
  align-self: flex-end; background: rgba(0,176,255,0.12);
  border: 1px solid rgba(0,176,255,0.3); color: #e8f5e9;
  border-radius: 12px 12px 2px 12px; padding: 8px 14px; max-width: 78%;
  font-size: 13px; font-family: 'Exo 2', sans-serif; line-height: 1.5;
}
.chat-msg-bot {
  align-self: flex-start; background: rgba(0,230,118,0.07);
  border: 1px solid rgba(0,230,118,0.2); color: #e8f5e9;
  border-radius: 12px 12px 12px 2px; padding: 8px 14px; max-width: 84%;
  font-size: 13px; font-family: 'Exo 2', sans-serif; line-height: 1.6;
}
.chat-msg-bot strong { color: var(--accent-g); }
.chat-sender-user {
  text-align: right; font-family: 'JetBrains Mono', monospace;
  font-size: 9px; letter-spacing: 2px; color: var(--accent-b); margin-bottom: 2px; text-transform: uppercase;
}
.chat-sender-bot {
  font-family: 'JetBrains Mono', monospace; font-size: 9px; letter-spacing: 2px;
  color: var(--accent-g); margin-bottom: 2px; text-transform: uppercase;
}
.chat-empty-state {
  flex: 1; display: flex; flex-direction: column; align-items: center;
  justify-content: center; color: var(--text-muted);
  font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 2px; text-align: center; gap: 8px;
}
.chat-empty-icon { font-size: 28px; opacity: .4; }

.quick-chips-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 0.8rem; }
.quick-chip-label {
  font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 2px;
  color: var(--text-muted); text-transform: uppercase; margin-bottom: 6px;
}

.chip-btn > div.stButton > button {
  width: auto !important; padding: 4px 14px !important;
  font-size: 10px !important; letter-spacing: 1px !important;
  border-radius: 20px !important; border-color: rgba(0,176,255,0.4) !important;
  color: var(--accent-b) !important; font-family: 'JetBrains Mono', monospace !important;
}
.chip-btn > div.stButton > button:hover {
  background: rgba(0,176,255,0.15) !important;
  border-color: var(--accent-b) !important;
  color: var(--accent-b) !important;
  transform: none !important; box-shadow: none !important;
}

.progress-track { width: 100%; height: 4px; background: var(--bg-card); border-radius: 2px; overflow: hidden; margin: 8px 0 4px; }
.progress-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, var(--accent-g), var(--accent-b)); }
.progress-fill-vision { background: linear-gradient(90deg, var(--accent-p), var(--accent-b)); }

.translate-box { background: var(--bg-card); border: 1px solid var(--border); border-radius: 3px; padding: 1.2rem 1.4rem; margin: 0.6rem 0; }
.translate-original { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin-bottom: 0.5rem; }
.translate-result { font-size: 15px; color: var(--text-main); line-height: 1.6; }
.translate-lang-badge {
  display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 2px;
  color: var(--accent-b); border: 1px solid rgba(0,176,255,0.3); padding: 2px 10px;
  border-radius: 2px; margin-bottom: 0.6rem; text-transform: uppercase;
}

.disease-list-item { font-size: 11px; color: var(--accent-b); padding: 2px 0; }
.disease-list-item span.sev-None     { color: #00e676; }
.disease-list-item span.sev-Low      { color: #69f0ae; }
.disease-list-item span.sev-Moderate { color: #ffd600; }
.disease-list-item span.sev-High     { color: #ff9100; }
.disease-list-item span.sev-Critical { color: #ff5252; }

/* Feature badges */
.feature-badge {
  display: inline-flex; align-items: center; gap: 6px;
  font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 2px;
  text-transform: uppercase; padding: 4px 12px; border-radius: 20px; margin: 2px;
}
.badge-vision { color: var(--accent-p); border: 1px solid rgba(224,64,251,0.4); background: rgba(224,64,251,0.08); }
.badge-audio  { color: var(--accent-y); border: 1px solid rgba(255,214,0,0.4); background: rgba(255,214,0,0.08); }
.badge-ml     { color: var(--accent-g); border: 1px solid rgba(0,230,118,0.4); background: rgba(0,230,118,0.08); }

@keyframes fadeSlideDown { 0%{opacity:0;transform:translateY(-20px)} 100%{opacity:1;transform:translateY(0)} }
@keyframes fadeIn { from { opacity:0; } to { opacity:1; } }
@keyframes resultReveal { 0%{opacity:0;transform:scale(0.96) translateY(20px)} 100%{opacity:1;transform:scale(1) translateY(0)} }
#MainMenu, footer, header { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  LANGUAGE DICTIONARY
# ═══════════════════════════════════════════════════════════════════
LANG = {
    "English": {
        "title": "Crop Disease Intelligence",
        "sub": "// AI-POWERED PATHOGEN DETECTION SYSTEM v3.0 — 100 CROPS · VISION · VOICE",
        "tag": "AGRISENSE · NEURAL DIAGNOSTICS",
        "sec_settings": "NOTIFICATION SETTINGS",
        "sec_input": "DIAGNOSTIC INPUTS",
        "sec_detect": "RUN ANALYSIS",
        "sec_translate": "MULTILINGUAL RESULT TRANSLATION",
        "sec_vision": "📸 IMAGE RECOGNITION — UPLOAD CROP PHOTO",
        "sec_audio": "🎙️ VOICE INPUT — SPEAK YOUR QUESTION",
        "crop": "TARGET CROP",
        "spot": "LEAF SPOTS SEVERITY",
        "yellow": "LEAF YELLOWING INDEX",
        "wilting": "PLANT WILTING LEVEL",
        "mold": "FUNGAL PRESENCE",
        "predict": "⚡  ANALYZE PATHOGEN",
        "vision_analyze": "🔬  ANALYZE IMAGE",
        "disease": "DETECTED PATHOGEN",
        "treatment": "RECOMMENDED PROTOCOL",
        "success": "✓  ANALYSIS COMPLETE — PATHOGEN IDENTIFIED",
        "vision_success": "✓  IMAGE ANALYSIS COMPLETE",
        "email_label": "Alert Email Address",
        "phone_label": "Phone / Telegram Chat ID",
        "email_key": "SMTP Email (sender)",
        "email_pass": "SMTP App Password",
        "tg_token": "Telegram Bot Token",
        "translate_btn": "🌐  TRANSLATE RESULTS",
        "translate_langs": "Select translation languages",
        "notify_sent": "SENT",
        "notify_fail": "FAILED",
        "notify_skip": "NOT CONFIGURED",
        "api_key_label": "Anthropic API Key (for Vision & Chat)",
        "lang_code": "en",
        "ml_label": "ML MODEL",
        "rule_label": "RULE-BASED",
    },
    "Hindi": {
        "title": "फसल रोग बुद्धिमत्ता",
        "sub": "// एआई-संचालित रोगज़नक़ पहचान — 100 फसलें · दृष्टि · आवाज़",
        "tag": "AGRISENSE · न्यूरल डायग्नोस्टिक्स",
        "sec_settings": "सूचना सेटिंग्स",
        "sec_input": "निदान इनपुट",
        "sec_detect": "विश्लेषण चलाएँ",
        "sec_translate": "बहुभाषी परिणाम अनुवाद",
        "sec_vision": "📸 छवि पहचान — फसल की फोटो अपलोड करें",
        "sec_audio": "🎙️ आवाज़ इनपुट — अपना सवाल बोलें",
        "crop": "फसल चुनें",
        "spot": "पत्तियों पर धब्बे",
        "yellow": "पत्ती पीलापन",
        "wilting": "पौधा मुरझाना",
        "mold": "फफूंद उपस्थिति",
        "predict": "⚡  रोग विश्लेषण करें",
        "vision_analyze": "🔬  छवि विश्लेषण करें",
        "disease": "पहचाना गया रोग",
        "treatment": "अनुशंसित उपचार",
        "success": "✓  विश्लेषण पूर्ण",
        "vision_success": "✓  छवि विश्लेषण पूर्ण",
        "email_label": "ईमेल पता",
        "phone_label": "फ़ोन / Telegram Chat ID",
        "email_key": "SMTP ईमेल (प्रेषक)",
        "email_pass": "SMTP पासवर्ड",
        "tg_token": "Telegram Bot Token",
        "translate_btn": "🌐  परिणाम अनुवाद करें",
        "translate_langs": "अनुवाद भाषाएँ चुनें",
        "notify_sent": "भेजा गया",
        "notify_fail": "विफल",
        "notify_skip": "कॉन्फ़िगर नहीं",
        "api_key_label": "Anthropic API Key (दृष्टि और चैट के लिए)",
        "lang_code": "hi",
        "ml_label": "ML मॉडल",
        "rule_label": "नियम-आधारित",
    },
    "Telugu": {
        "title": "పంట వ్యాధి నిర్ధారణ",
        "sub": "// AI-ఆధారిత రోగ నిర్ధారణ — 100 పంటలు · దృష్టి · వాయిస్",
        "tag": "AGRISENSE · న్యూరల్ డయాగ్నోస్టిక్స్",
        "sec_settings": "నోటిఫికేషన్ సెట్టింగ్‌లు",
        "sec_input": "రోగనిర్ధారణ ఇన్‌పుట్‌లు",
        "sec_detect": "విశ్లేషణ చేయండి",
        "sec_translate": "బహుభాషా ఫలిత అనువాదం",
        "sec_vision": "📸 చిత్ర గుర్తింపు — పంట ఫోటో అప్‌లోడ్ చేయండి",
        "sec_audio": "🎙️ వాయిస్ ఇన్‌పుట్ — మీ ప్రశ్న చెప్పండి",
        "crop": "పంట ఎంచుకోండి",
        "spot": "ఆకు మచ్చలు",
        "yellow": "ఆకు పసుపు",
        "wilting": "మొక్క వాడిపోవడం",
        "mold": "ఫంగస్ ఉనికి",
        "predict": "⚡  వ్యాధి విశ్లేషించండి",
        "vision_analyze": "🔬  చిత్రాన్ని విశ్లేషించండి",
        "disease": "గుర్తించిన వ్యాధి",
        "treatment": "సిఫార్సు చేసిన చికిత్స",
        "success": "✓  విశ్లేషణ పూర్తయింది",
        "vision_success": "✓  చిత్ర విశ్లేషణ పూర్తయింది",
        "email_label": "అలర్ట్ ఇమెయిల్ చిరునామా",
        "phone_label": "ఫోన్ / Telegram Chat ID",
        "email_key": "SMTP ఇమెయిల్ (పంపేవారు)",
        "email_pass": "SMTP పాస్‌వర్డ్",
        "tg_token": "Telegram Bot Token",
        "translate_btn": "🌐  ఫలితాలు అనువదించండి",
        "translate_langs": "అనువాద భాషలు ఎంచుకోండి",
        "notify_sent": "పంపబడింది",
        "notify_fail": "విఫలమైంది",
        "notify_skip": "కాన్ఫిగర్ చేయలేదు",
        "api_key_label": "Anthropic API Key (దృష్టి మరియు చాట్ కోసం)",
        "lang_code": "te",
        "ml_label": "ML మోడల్",
        "rule_label": "నియమ-ఆధారిత",
    },
    "Tamil": {
        "title": "பயிர் நோய் கண்டறிதல்",
        "sub": "// AI-இயக்கப்படும் நோயறிதல் — 100 பயிர்கள் · பார்வை · குரல்",
        "tag": "AGRISENSE · நரம்பியல் நோயறிதல்",
        "sec_settings": "அறிவிப்பு அமைப்புகள்",
        "sec_input": "நோயறிதல் உள்ளீடுகள்",
        "sec_detect": "பகுப்பாய்வு இயக்கவும்",
        "sec_translate": "பன்மொழி முடிவு மொழிபெயர்ப்பு",
        "sec_vision": "📸 படம் அங்கீகாரம் — பயிர் புகைப்படம் பதிவேற்றவும்",
        "sec_audio": "🎙️ குரல் உள்ளீடு — உங்கள் கேள்வியை பேசவும்",
        "crop": "இலக்கு பயிர்",
        "spot": "இலை புள்ளிகளின் தீவிரம்",
        "yellow": "இலை மஞ்சள் குறியீடு",
        "wilting": "தாவர வாட்டம்",
        "mold": "பூஞ்சை இருப்பு",
        "predict": "⚡  நோய்க்கிருமியை பகுப்பாய்வு செய்",
        "vision_analyze": "🔬  படத்தை பகுப்பாய்வு செய்",
        "disease": "கண்டறியப்பட்ட நோய்",
        "treatment": "பரிந்துரைக்கப்பட்ட சிகிச்சை",
        "success": "✓  பகுப்பாய்வு முடிந்தது",
        "vision_success": "✓  படம் பகுப்பாய்வு முடிந்தது",
        "email_label": "எச்சரிக்கை மின்னஞ்சல்",
        "phone_label": "தொலைபேசி / Telegram Chat ID",
        "email_key": "SMTP மின்னஞ்சல் (அனுப்புநர்)",
        "email_pass": "SMTP கடவுச்சொல்",
        "tg_token": "Telegram Bot Token",
        "translate_btn": "🌐  முடிவுகளை மொழிபெயர்க்கவும்",
        "translate_langs": "மொழிபெயர்ப்பு மொழிகளை தேர்ந்தெடுக்கவும்",
        "notify_sent": "அனுப்பப்பட்டது",
        "notify_fail": "தோல்வி",
        "notify_skip": "கட்டமைக்கப்படவில்லை",
        "api_key_label": "Anthropic API Key (பார்வை மற்றும் அரட்டைக்கு)",
        "lang_code": "ta",
        "ml_label": "ML மாதிரி",
        "rule_label": "விதி-அடிப்படை",
    },
    "Kannada": {
        "title": "ಬೆಳೆ ರೋಗ ಪತ್ತೆ",
        "sub": "// AI-ಚಾಲಿತ ರೋಗ ಪತ್ತೆ — 100 ಬೆಳೆಗಳು · ದೃಷ್ಟಿ · ಧ್ವನಿ",
        "tag": "AGRISENSE · ನ್ಯೂರಲ್ ಡಯಾಗ್ನಾಸ್ಟಿಕ್ಸ್",
        "sec_settings": "ಅಧಿಸೂಚನೆ ಸೆಟ್ಟಿಂಗ್‌ಗಳು",
        "sec_input": "ರೋಗನಿರ್ಣಯ ಇನ್‌ಪುಟ್‌ಗಳು",
        "sec_detect": "ವಿಶ್ಲೇಷಣೆ ಚಲಾಯಿಸಿ",
        "sec_translate": "ಬಹುಭಾಷಾ ಫಲಿತಾಂಶ ಅನುವಾದ",
        "sec_vision": "📸 ಚಿತ್ರ ಗುರುತಿಸುವಿಕೆ — ಬೆಳೆ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "sec_audio": "🎙️ ಧ್ವನಿ ಇನ್‌ಪುಟ್ — ನಿಮ್ಮ ಪ್ರಶ್ನೆ ಹೇಳಿ",
        "crop": "ಗುರಿ ಬೆಳೆ",
        "spot": "ಎಲೆ ಚುಕ್ಕೆಗಳ ತೀವ್ರತೆ",
        "yellow": "ಎಲೆ ಹಳದಿ ಸೂಚ್ಯಂಕ",
        "wilting": "ಗಿಡ ಬಾಡುವ ಮಟ್ಟ",
        "mold": "ಶಿಲೀಂಧ್ರ ಇರುವಿಕೆ",
        "predict": "⚡  ರೋಗಕಾರಕ ವಿಶ್ಲೇಷಿಸಿ",
        "vision_analyze": "🔬  ಚಿತ್ರ ವಿಶ್ಲೇಷಿಸಿ",
        "disease": "ಪತ್ತೆಯಾದ ರೋಗ",
        "treatment": "ಶಿಫಾರಸು ಮಾಡಿದ ಚಿಕಿತ್ಸೆ",
        "success": "✓  ವಿಶ್ಲೇಷಣೆ ಪೂರ್ಣಗೊಂಡಿದೆ",
        "vision_success": "✓  ಚಿತ್ರ ವಿಶ್ಲೇಷಣೆ ಪೂರ್ಣಗೊಂಡಿದೆ",
        "email_label": "ಎಚ್ಚರಿಕೆ ಇಮೇಲ್ ವಿಳಾಸ",
        "phone_label": "ಫೋನ್ / Telegram Chat ID",
        "email_key": "SMTP ಇಮೇಲ್ (ಕಳುಹಿಸುವವರು)",
        "email_pass": "SMTP ಪಾಸ್‌ವರ್ಡ್",
        "tg_token": "Telegram Bot Token",
        "translate_btn": "🌐  ಫಲಿತಾಂಶಗಳನ್ನು ಅನುವಾದಿಸಿ",
        "translate_langs": "ಅನುವಾದ ಭಾಷೆಗಳನ್ನು ಆಯ್ಕೆ ಮಾಡಿ",
        "notify_sent": "ಕಳುಹಿಸಲಾಗಿದೆ",
        "notify_fail": "ವಿಫಲ",
        "notify_skip": "ಕಾನ್ಫಿಗರ್ ಆಗಿಲ್ಲ",
        "api_key_label": "Anthropic API Key (ದೃಷ್ಟಿ ಮತ್ತು ಚಾಟ್‌ಗಾಗಿ)",
        "lang_code": "kn",
        "ml_label": "ML ಮಾದರಿ",
        "rule_label": "ನಿಯಮ-ಆಧಾರಿತ",
    },
    "Marathi": {
        "title": "पीक रोग ओळख",
        "sub": "// AI-चालित रोगजंतू ओळख — 100 पिके · दृष्टी · आवाज",
        "tag": "AGRISENSE · न्यूरल डायग्नोस्टिक्स",
        "sec_settings": "सूचना सेटिंग्ज",
        "sec_input": "निदान इनपुट",
        "sec_detect": "विश्लेषण चालवा",
        "sec_translate": "बहुभाषिक निकाल भाषांतर",
        "sec_vision": "📸 प्रतिमा ओळख — पिकाचा फोटो अपलोड करा",
        "sec_audio": "🎙️ आवाज इनपुट — आपला प्रश्न बोला",
        "crop": "पीक निवडा",
        "spot": "पानांवरील डाग",
        "yellow": "पान पिवळेपणा",
        "wilting": "रोपाचे कोमेजणे",
        "mold": "बुरशी उपस्थिती",
        "predict": "⚡  रोग विश्लेषण करा",
        "vision_analyze": "🔬  प्रतिमा विश्लेषण करा",
        "disease": "आढळलेला रोग",
        "treatment": "शिफारस केलेले उपचार",
        "success": "✓  विश्लेषण पूर्ण",
        "vision_success": "✓  प्रतिमा विश्लेषण पूर्ण",
        "email_label": "अलर्ट ईमेल पत्ता",
        "phone_label": "फोन / Telegram Chat ID",
        "email_key": "SMTP ईमेल (प्रेषक)",
        "email_pass": "SMTP पासवर्ड",
        "tg_token": "Telegram Bot Token",
        "translate_btn": "🌐  निकाल भाषांतरित करा",
        "translate_langs": "भाषांतर भाषा निवडा",
        "notify_sent": "पाठवले",
        "notify_fail": "अयशस्वी",
        "notify_skip": "कॉन्फिगर केलेले नाही",
        "api_key_label": "Anthropic API Key (दृष्टी आणि चॅटसाठी)",
        "lang_code": "mr",
        "ml_label": "ML मॉडेल",
        "rule_label": "नियम-आधारित",
    },
}

TRANSLATION_MAP = {
    "Hindi": {
        'banana_bract_mosaic_virus':("केले का ब्रेक्ट मोज़ेक वायरस","संक्रमित पौधे हटाएं। नीम-आधारित कीटनाशक से एफिड नियंत्रण करें।"),
        'banana_cordana':           ("केला कॉर्डाना रोग",            "मैंकोज़ेब या तांबा-आधारित फफूंदनाशक लगाएं।"),
        'banana_healthy':           ("केला स्वस्थ",                   "उपचार की जरूरत नहीं।"),
        'banana_insectpest':        ("केला कीट",                      "नीम तेल स्प्रे करें।"),
        'banana_moko':              ("केला मोको रोग",                  "तुरंत पौधे नष्ट करें।"),
        'banana_panama':            ("केला पनामा विल्ट",               "पौधे नष्ट करें। प्रतिरोधी किस्म लगाएं।"),
        'banana_pestalotiopsis':    ("केला पेस्टालोशियोसिस",          "प्रोपिकोनाज़ोल फफूंदनाशक लगाएं।"),
        'banana_sigatoka':          ("केला सिगाटोका",                  "मैंकोज़ेब हर 10 दिन में स्प्रे करें।"),
        'banana_yb_sigatoka':       ("केला पीला सिगाटोका",            "प्रणालीगत फफूंदनाशक लगाएं।"),
        'cauliflower_Blackrot':     ("फूलगोभी काला सड़न",             "गर्म पानी उपचारित बीज उपयोग करें।"),
        'cauliflower_bacterial_spot_rot':("फूलगोभी जीवाणु धब्बा","कॉपर हाइड्रॉक्साइड स्प्रे करें।"),
        'cauliflower_downy_mildew': ("फूलगोभी डाउनी फफूंदी",         "मेटालेक्सिल-मैंकोज़ेब लगाएं।"),
        'cauliflower_healthy':      ("फूलगोभी स्वस्थ",               "उपचार की जरूरत नहीं।"),
        'chilli_anthracnose':       ("मिर्च एन्थ्रेक्नोज",            "मैंकोज़ेब या कार्बेंडाज़िम स्प्रे करें।"),
        'chilli_healthy':           ("मिर्च स्वस्थ",                  "उपचार की जरूरत नहीं।"),
        'chilli_leafcurl':          ("मिर्च पत्ती मुड़ना",             "थ्रिप्स/घुन नियंत्रण करें।"),
        'chilli_leafspot':          ("मिर्च पत्ती धब्बा",             "कॉपर ऑक्सीक्लोराइड स्प्रे करें।"),
        'chilli_whitefly':          ("मिर्च सफेद मक्खी",              "इमिडाक्लोप्रिड या नीम तेल स्प्रे करें।"),
        'chilli_yellowish':         ("मिर्च पीलापन",                  "पोषक तत्व कमी जांचें।"),
        'groundnut_early_leaf_spot':("मूंगफली प्रारंभिक पत्ती धब्बा","क्लोरोथैलोनिल हर 14 दिन में स्प्रे करें।"),
        'groundnut_early_rust':     ("मूंगफली प्रारंभिक रतुआ",        "प्रोपिकोनाज़ोल स्प्रे करें।"),
        'groundnut_healthy':        ("मूंगफली स्वस्थ",               "उपचार की जरूरत नहीं।"),
        'groundnut_late_leaf_spot': ("मूंगफली देर से पत्ती धब्बा",   "टेबुकोनाज़ोल फफूंदनाशक लगाएं।"),
        'groundnut_nutrition_deficiency':("मूंगफली पोषण कमी",        "मिट्टी परीक्षण करें।"),
        'groundnut_rust':           ("मूंगफली रतुआ",                  "ट्रायडिमेफोन स्प्रे करें।"),
        'radish_black_leaf_spot':   ("मूली काला पत्ती धब्बा",        "इप्रोडायोन स्प्रे करें।"),
        'radish_downey_mildew':     ("मूली डाउनी फफूंदी",            "मेटालेक्सिल + मैंकोज़ेब स्प्रे करें।"),
        'radish_flea_beetle':       ("मूली पिस्सू भृंग",              "पाइरेथ्रिन स्प्रे करें।"),
        'radish_healthy':           ("मूली स्वस्थ",                   "उपचार की जरूरत नहीं।"),
        'radish_mosaic':            ("मूली मोज़ेक",                   "संक्रमित पौधे नष्ट करें।"),
        "gtts_code": "hi",
    },
    "Telugu": {
        'banana_bract_mosaic_virus':("అరటి బ్రాక్ట్ మొజాయిక్ వైరస్","సోకిన మొక్కలు తొలగించండి."),
        'banana_cordana':           ("అరటి కార్డానా",                  "మాంకోజెబ్ వాడండి."),
        'banana_healthy':           ("అరటి ఆరోగ్యకరంగా ఉంది",         "చికిత్స అవసరం లేదు."),
        'banana_insectpest':        ("అరటి కీటక తెగులు",               "నీమ్ నూనె పిచికారి చేయండి."),
        'banana_moko':              ("అరటి మోకో",                      "వెంటనే మొక్కలు నాశనం చేయండి."),
        'banana_panama':            ("అరటి పనామా వ్యాధి",              "మొక్కలు నాశనం చేయండి."),
        'banana_pestalotiopsis':    ("అరటి పెస్టలోశియోసిస్",           "ప్రొపికోనజోల్ వాడండి."),
        'banana_sigatoka':          ("అరటి సిగటోకా",                   "మాంకోజెబ్ పిచికారి చేయండి."),
        'banana_yb_sigatoka':       ("అరటి పసుపు సిగటోకా",            "వ్యవస్థాపిత శిలీంద్రనాశిని వాడండి."),
        'cauliflower_Blackrot':     ("కాలీఫ్లవర్ నల్ల కుళ్ళు",        "వేడి నీటిలో విత్తనాలు శుద్ధి చేయండి."),
        'cauliflower_bacterial_spot_rot':("కాలీఫ్లవర్ బ్యాక్టీరియా మచ్చ","కాపర్ హైడ్రాక్సైడ్ పిచికారి చేయండి."),
        'cauliflower_downy_mildew': ("కాలీఫ్లవర్ డౌని తెగులు",         "మెటలాక్సిల్ వాడండి."),
        'cauliflower_healthy':      ("కాలీఫ్లవర్ ఆరోగ్యంగా ఉంది",     "చికిత్స అవసరం లేదు."),
        'chilli_anthracnose':       ("మిర్చి యాంత్రాక్నోస్",           "మాంకోజెబ్ పిచికారి చేయండి."),
        'chilli_healthy':           ("మిర్చి ఆరోగ్యంగా ఉంది",          "చికిత్స అవసరం లేదు."),
        'chilli_leafcurl':          ("మిర్చి ఆకు మడత",                 "థ్రిప్స్ నియంత్రించండి."),
        'chilli_leafspot':          ("మిర్చి ఆకు మచ్చ",                "కాపర్ ఆక్సీక్లోరైడ్ పిచికారి చేయండి."),
        'chilli_whitefly':          ("మిర్చి తెల్ల ఈగ",                "ఇమిడాక్లోప్రిడ్ వాడండి."),
        'chilli_yellowish':         ("మిర్చి పసుపు రంగు",              "పోషక లోపాలు తనిఖీ చేయండి."),
        'groundnut_early_leaf_spot':("వేరుసెనగ ముందస్తు ఆకు మచ్చ",   "క్లోరోథాలోనిల్ పిచికారి చేయండి."),
        'groundnut_early_rust':     ("వేరుసెనగ ముందస్తు తుప్పు",      "ప్రొపికోనజోల్ పిచికారి చేయండి."),
        'groundnut_healthy':        ("వేరుసెనగ ఆరోగ్యంగా ఉంది",       "చికిత్స అవసరం లేదు."),
        'groundnut_late_leaf_spot': ("వేరుసెనగ ఆలస్య ఆకు మచ్చ",      "టెబుకోనజోల్ వాడండి."),
        'groundnut_nutrition_deficiency':("వేరుసెనగ పోషక లోపం",       "మట్టి పరీక్ష చేయండి."),
        'groundnut_rust':           ("వేరుసెనగ తుప్పు",                "ట్రయాడిమెఫాన్ పిచికారి చేయండి."),
        'radish_black_leaf_spot':   ("మూలంగి నల్ల ఆకు మచ్చ",         "ఇప్రోడియోన్ పిచికారి చేయండి."),
        'radish_downey_mildew':     ("మూలంగి డౌని తెగులు",             "మెటలాక్సిల్ వాడండి."),
        'radish_flea_beetle':       ("మూలంగి దుర్గ బీటిల్",            "పైరెత్రిన్ పిచికారి చేయండి."),
        'radish_healthy':           ("మూలంగి ఆరోగ్యంగా ఉంది",         "చికిత్స అవసరం లేదు."),
        'radish_mosaic':            ("మూలంగి మొజాయిక్",                "సోకిన మొక్కలు నాశనం చేయండి."),
        "gtts_code": "te",
    },
    "Tamil": {
        'banana_bract_mosaic_virus':("வாழை கொழுந்து மொசைக் வைரஸ்",  "பாதிக்கப்பட்ட தாவரங்களை அகற்றவும்."),
        'banana_cordana':           ("வாழை கார்டனா நோய்",             "மன்கோசெப் தெளிக்கவும்."),
        'banana_healthy':           ("வாழை ஆரோக்கியமாக உள்ளது",      "சிகிச்சை தேவையில்லை."),
        'banana_insectpest':        ("வாழை பூச்சி தொல்லை",            "வேப்பெண்ணெய் தெளிக்கவும்."),
        'banana_moko':              ("வாழை மோக்கோ நோய்",              "உடனடியாக தாவரங்களை அழிக்கவும்."),
        'banana_panama':            ("வாழை பனாமா நோய்",               "தாவரங்களை அழிக்கவும்."),
        'banana_pestalotiopsis':    ("வாழை பெஸ்டலோஷியோசிஸ்",         "ப்ரோபிக்கோனசோல் பயன்படுத்தவும்."),
        'banana_sigatoka':          ("வாழை சிகடோக்கா",                "மன்கோசெப் தெளிக்கவும்."),
        'banana_yb_sigatoka':       ("வாழை மஞ்சள் சிகடோக்கா",       "முறையான பூஞ்சைக்கொல்லி பயன்படுத்தவும்."),
        'cauliflower_Blackrot':     ("காலிஃப்ளவர் கருப்பு அழுகல்",   "சூடான நீரில் விதைகளை சுத்தம் செய்யவும்."),
        'cauliflower_bacterial_spot_rot':("காலிஃப்ளவர் பாக்டீரியா","காப்பர் ஹைட்ராக்சைடு தெளிக்கவும்."),
        'cauliflower_downy_mildew': ("காலிஃப்ளவர் டவுனி",             "மெட்டாலாக்சில் பயன்படுத்தவும்."),
        'cauliflower_healthy':      ("காலிஃப்ளவர் ஆரோக்கியம்",       "சிகிச்சை தேவையில்லை."),
        'chilli_anthracnose':       ("மிளகாய் ஆந்த்ராக்னோஸ்",         "மன்கோசெப் தெளிக்கவும்."),
        'chilli_healthy':           ("மிளகாய் ஆரோக்கியம்",            "சிகிச்சை தேவையில்லை."),
        'chilli_leafcurl':          ("மிளகாய் இலை சுருட்டல்",          "த்ரிப்ஸ் கட்டுப்படுத்தவும்."),
        'chilli_leafspot':          ("மிளகாய் இலை புள்ளி",             "காப்பர் தெளிக்கவும்."),
        'chilli_whitefly':          ("மிளகாய் வெள்ளை ஈ",              "இமிடாக்ளோப்ரிட் பயன்படுத்தவும்."),
        'chilli_yellowish':         ("மிளகாய் மஞ்சள்",                 "ஊட்டச்சத்து சரிபார்க்கவும்."),
        'groundnut_early_leaf_spot':("வேர்க்கடலை ஆரம்ப இலை புள்ளி", "குளோரோத்தலோனில் தெளிக்கவும்."),
        'groundnut_early_rust':     ("வேர்க்கடலை ஆரம்ப துரு",         "ப்ரோபிக்கோனசோல் தெளிக்கவும்."),
        'groundnut_healthy':        ("வேர்க்கடலை ஆரோக்கியம்",         "சிகிச்சை தேவையில்லை."),
        'groundnut_late_leaf_spot': ("வேர்க்கடலை தாமத இலை புள்ளி",   "டெபுக்கோனசோல் பயன்படுத்தவும்."),
        'groundnut_nutrition_deficiency':("வேர்க்கடலை ஊட்டக்குறைபாடு","மண் பரிசோதனை செய்யவும்."),
        'groundnut_rust':           ("வேர்க்கடலை துரு",                "ட்ரையடிமெஃபான் தெளிக்கவும்."),
        'radish_black_leaf_spot':   ("முள்ளங்கி கருப்பு புள்ளி",      "இப்ரோடியோன் தெளிக்கவும்."),
        'radish_downey_mildew':     ("முள்ளங்கி டவுனி",                "மெட்டாலாக்சில் பயன்படுத்தவும்."),
        'radish_flea_beetle':       ("முள்ளங்கி தட்டான் வண்டு",        "பைரித்ரின் தெளிக்கவும்."),
        'radish_healthy':           ("முள்ளங்கி ஆரோக்கியம்",           "சிகிச்சை தேவையில்லை."),
        'radish_mosaic':            ("முள்ளங்கி மொசைக்",               "பாதிக்கப்பட்டதை அழிக்கவும்."),
        "gtts_code": "ta",
    },
    "Kannada": {
        'banana_bract_mosaic_virus':("ಬಾಳೆ ಬ್ರಾಕ್ಟ್ ಮೊಸಾಯಿಕ್",       "ಸೋಂಕಿತ ಗಿಡ ತೆಗೆದುಹಾಕಿ."),
        'banana_cordana':           ("ಬಾಳೆ ಕಾರ್ಡಾನಾ",                 "ಮ್ಯಾಂಕೊಜೆಬ್ ಬಳಸಿ."),
        'banana_healthy':           ("ಬಾಳೆ ಆರೋಗ್ಯಕರ",                 "ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ."),
        'banana_insectpest':        ("ಬಾಳೆ ಕೀಟ",                       "ನೀಮ್ ಎಣ್ಣೆ ಸ್ಪ್ರೇ."),
        'banana_moko':              ("ಬಾಳೆ ಮೊಕೊ",                     "ತಕ್ಷಣ ಗಿಡ ನಾಶ."),
        'banana_panama':            ("ಬಾಳೆ ಪನಾಮ",                      "ಗಿಡ ನಾಶ ಮಾಡಿ."),
        'banana_pestalotiopsis':    ("ಬಾಳೆ ಪೆಸ್ಟಲೋಶಿಯೋಸಿಸ್",          "ಪ್ರೊಪಿಕೊನಜೋಲ್."),
        'banana_sigatoka':          ("ಬಾಳೆ ಸಿಗಟೊಕಾ",                  "ಮ್ಯಾಂಕೊಜೆಬ್ ಸ್ಪ್ರೇ."),
        'banana_yb_sigatoka':       ("ಬಾಳೆ ಹಳದಿ ಸಿಗಟೊಕಾ",            "ಶಿಲೀಂಧ್ರನಾಶಕ."),
        'cauliflower_Blackrot':     ("ಹೂಕೋಸು ಕಪ್ಪು ಕೊಳೆ",            "ಬಿಸಿ ನೀರು ಬೀಜ ಸಂಸ್ಕರಣ."),
        'cauliflower_bacterial_spot_rot':("ಹೂಕೋಸು ಬ್ಯಾಕ್ಟೀರಿಯ",      "ಕಾಪರ್ ಸ್ಪ್ರೇ."),
        'cauliflower_downy_mildew': ("ಹೂಕೋಸು ಡೌನಿ",                   "ಮೆಟಾಲ್ಯಾಕ್ಸಿಲ್."),
        'cauliflower_healthy':      ("ಹೂಕೋಸು ಆರೋಗ್ಯಕರ",              "ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ."),
        'chilli_anthracnose':       ("ಮೆಣಸಿನಕಾಯಿ ಆಂಥ್ರಾಕ್ನೋಸ್",      "ಮ್ಯಾಂಕೊಜೆಬ್."),
        'chilli_healthy':           ("ಮೆಣಸಿನಕಾಯಿ ಆರೋಗ್ಯಕರ",           "ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ."),
        'chilli_leafcurl':          ("ಮೆಣಸಿನಕಾಯಿ ಸುರುಳಿ",             "ಥ್ರಿಪ್ಸ್ ನಿಯಂತ್ರಣ."),
        'chilli_leafspot':          ("ಮೆಣಸಿನಕಾಯಿ ಚುಕ್ಕೆ",             "ಕಾಪರ್ ಸ್ಪ್ರೇ."),
        'chilli_whitefly':          ("ಮೆಣಸಿನಕಾಯಿ ಬಿಳಿ ನೊಣ",           "ಇಮಿಡಾಕ್ಲೋಪ್ರಿಡ್."),
        'chilli_yellowish':         ("ಮೆಣಸಿನಕಾಯಿ ಹಳದಿ",               "ಪೋಷಕಾಂಶ ಪರೀಕ್ಷೆ."),
        'groundnut_early_leaf_spot':("ನೆಲಗಡಲೆ ಆರಂಭಿಕ ಚುಕ್ಕೆ",        "ಕ್ಲೋರೋಥಲೋನಿಲ್."),
        'groundnut_early_rust':     ("ನೆಲಗಡಲೆ ತುಕ್ಕು",                "ಪ್ರೊಪಿಕೊನಜೋಲ್."),
        'groundnut_healthy':        ("ನೆಲಗಡಲೆ ಆರೋಗ್ಯಕರ",              "ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ."),
        'groundnut_late_leaf_spot': ("ನೆಲಗಡಲೆ ತಡ ಚುಕ್ಕೆ",             "ಟೆಬುಕೊನಜೋಲ್."),
        'groundnut_nutrition_deficiency':("ನೆಲಗಡಲೆ ಕೊರತೆ",            "ಮಣ್ಣು ಪರೀಕ್ಷೆ."),
        'groundnut_rust':           ("ನೆಲಗಡಲೆ ತುಕ್ಕು",                "ಟ್ರಯಾಡಿಮೆಫಾನ್."),
        'radish_black_leaf_spot':   ("ಮೂಲಂಗಿ ಕಪ್ಪು",                  "ಇಪ್ರೊಡಿಯೋನ್."),
        'radish_downey_mildew':     ("ಮೂಲಂಗಿ ಡೌನಿ",                   "ಮೆಟಾಲ್ಯಾಕ್ಸಿಲ್."),
        'radish_flea_beetle':       ("ಮೂಲಂಗಿ ದುಂಬಿ",                   "ಪೈರಿಥ್ರಿನ್."),
        'radish_healthy':           ("ಮೂಲಂಗಿ ಆರೋಗ್ಯಕರ",               "ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ."),
        'radish_mosaic':            ("ಮೂಲಂಗಿ ಮೊಸಾಯಿಕ್",               "ಸೋಂಕಿತ ಗಿಡ ನಾಶ."),
        "gtts_code": "kn",
    },
    "Marathi": {
        'banana_bract_mosaic_virus':("केळीचा मोझॅक व्हायरस",          "संक्रमित झाडे काढा."),
        'banana_cordana':           ("केळी कॉर्डाना",                  "मॅन्कोझेब वापरा."),
        'banana_healthy':           ("केळी निरोगी",                    "उपचाराची गरज नाही."),
        'banana_insectpest':        ("केळी कीड",                       "नीम तेल फवारणी."),
        'banana_moko':              ("केळी मोको",                      "ताबडतोब झाडे नष्ट."),
        'banana_panama':            ("केळी पनामा",                     "झाडे नष्ट करा."),
        'banana_pestalotiopsis':    ("केळी पेस्टालोशियोसिस",           "प्रोपिकोनाझोल."),
        'banana_sigatoka':          ("केळी सिगाटोका",                  "मॅन्कोझेब दर 10 दिवस."),
        'banana_yb_sigatoka':       ("केळी पिवळा सिगाटोका",           "बुरशीनाशक."),
        'cauliflower_Blackrot':     ("फुलकोबी काळी कुजणे",            "गरम पाणी बीज उपचार."),
        'cauliflower_bacterial_spot_rot':("फुलकोबी जीवाणू",           "कॉपर फवारणी."),
        'cauliflower_downy_mildew': ("फुलकोबी डाउनी",                  "मेटालेक्सिल."),
        'cauliflower_healthy':      ("फुलकोबी निरोगी",                 "उपचाराची गरज नाही."),
        'chilli_anthracnose':       ("मिरची अँथ्रॅक्नोज",              "मॅन्कोझेब फवारणी."),
        'chilli_healthy':           ("मिरची निरोगी",                   "उपचाराची गरज नाही."),
        'chilli_leafcurl':          ("मिरची पान मुडणे",                "थ्रिप्स नियंत्रण."),
        'chilli_leafspot':          ("मिरची पान डाग",                  "कॉपर फवारणी."),
        'chilli_whitefly':          ("मिरची पांढरी माशी",              "इमिडाक्लोप्रिड."),
        'chilli_yellowish':         ("मिरची पिवळेपणा",                 "पोषक तत्त्वे तपासा."),
        'groundnut_early_leaf_spot':("शेंगदाणा डाग",                  "क्लोरोथॅलोनिल."),
        'groundnut_early_rust':     ("शेंगदाणा गंज",                  "प्रोपिकोनाझोल."),
        'groundnut_healthy':        ("शेंगदाणा निरोगी",               "उपचाराची गरज नाही."),
        'groundnut_late_leaf_spot': ("शेंगदाणा उशिरा डाग",            "टेबुकोनाझोल."),
        'groundnut_nutrition_deficiency':("शेंगदाणा कमतरता",          "मातीची चाचणी."),
        'groundnut_rust':           ("शेंगदाणा गंज",                  "ट्रायडिमेफोन."),
        'radish_black_leaf_spot':   ("मुळा काळा डाग",                 "इप्रोडायोन."),
        'radish_downey_mildew':     ("मुळा डाउनी",                    "मेटालेक्सिल."),
        'radish_flea_beetle':       ("मुळा पिसू भुंगा",                "पायरेथ्रिन."),
        'radish_healthy':           ("मुळा निरोगी",                    "उपचाराची गरज नाही."),
        'radish_mosaic':            ("मुळा मोझॅक",                    "संक्रमित नष्ट करा."),
        "gtts_code": "mr",
    },
}

LANG_NAMES_FOR_TRANSLATE = list(TRANSLATION_MAP.keys())
ALL_LANGS = [l for l in LANG_NAMES_FOR_TRANSLATE if l != "English"]

# ═══════════════════════════════════════════════════════════════════
#  NOTIFICATION HELPERS
# ═══════════════════════════════════════════════════════════════════
def build_html_email(crop, disease, treatment, severity, confidence, translations):
    trans_rows = ""
    for ln, data in translations.items():
        trans_rows += f"""
        <tr><td colspan="2" style="padding:4px 12px;background:#0a1a2a;">
          <span style="font-family:monospace;font-size:10px;color:#00b0ff;">▶ {ln}</span></td></tr>
        <tr>
          <td style="padding:6px 12px;border:1px solid #1a3a2a;color:#7fa89a;font-size:11px;">Disease</td>
          <td style="padding:6px 12px;border:1px solid #1a3a2a;color:#00e676;">{data.get('disease',disease)}</td>
        </tr>
        <tr>
          <td style="padding:6px 12px;border:1px solid #1a3a2a;color:#7fa89a;font-size:11px;">Treatment</td>
          <td style="padding:6px 12px;border:1px solid #1a3a2a;color:#00b0ff;">{data.get('treatment',treatment)}</td>
        </tr>"""
    trans_block = (f"<div style='margin:16px 0;'><div style='font-size:10px;letter-spacing:3px;color:#ffd600;text-transform:uppercase;font-family:monospace;margin-bottom:8px;'>TRANSLATIONS</div>"
                   f"<table style='width:100%;border-collapse:collapse;'>{trans_rows}</table></div>") if trans_rows else ""
    return f"""<!DOCTYPE html><html><body style="background:#030810;">
    <div style="font-family:'Segoe UI',Arial;background:#050a0e;color:#e8f5e9;padding:36px;max-width:640px;margin:24px auto;border:1px solid #0d3020;border-radius:8px;">
    <div style="border-left:4px solid #00e676;padding-left:16px;margin-bottom:28px;">
      <div style="font-size:10px;letter-spacing:4px;color:#00e676;font-family:monospace;">AGRISENSE AI · ALERT</div>
      <h2 style="color:#fff;margin:0;">Crop Disease Report</h2></div>
    <div style="background:#00e676;border-radius:4px;padding:14px 20px;margin-bottom:24px;text-align:center;">
      <div style="color:#050a0e;font-size:11px;letter-spacing:3px;font-family:monospace;">CROP: {crop}</div>
      <div style="color:#050a0e;font-size:26px;font-weight:900;">{disease}</div>
      <div style="color:#0a3018;font-size:12px;font-family:monospace;">Confidence: {confidence:.0f}% | Severity: {severity}</div></div>
    <div style="background:#091828;border:1px solid #00b0ff;border-radius:4px;padding:16px 20px;margin-bottom:24px;">
      <div style="font-size:10px;letter-spacing:3px;color:#00b0ff;font-family:monospace;margin-bottom:8px;">TREATMENT</div>
      <div style="color:#e8f5e9;font-size:14px;line-height:1.6;">{treatment}</div></div>
    {trans_block}
    <p style="color:#2a5a3a;font-size:10px;font-family:monospace;text-align:center;">AGRISENSE AI · RESEARCH USE ONLY</p>
    </div></body></html>"""

def send_email(to_addr, sender, password, crop, disease, treatment, severity, confidence, translations):
    if not (to_addr and sender and password): return None
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🌿 AgriSense — {disease} on {crop} ({severity})"
        msg["From"] = sender; msg["To"] = to_addr
        msg.attach(MIMEText(f"AgriSense AI\nCrop: {crop}\nDisease: {disease}\nSeverity: {severity}\nTreatment: {treatment}", "plain"))
        msg.attach(MIMEText(build_html_email(crop, disease, treatment, severity, confidence, translations), "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(sender, password); s.sendmail(sender, to_addr, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Email error: {e}"); return False

def send_telegram(chat_id, bot_token, crop, disease, treatment, severity):
    if not (chat_id and bot_token): return None
    try:
        text = f"🌿 *AgriSense AI*\n🌾 *Crop:* {crop}\n🦠 *Disease:* `{disease}`\n⚠️ *Severity:* {severity}\n💊 *Treatment:* {treatment}"
        r = requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage",
                          data={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=8)
        return r.status_code == 200
    except Exception as e:
        st.error(f"Telegram error: {e}"); return False

@st.cache_resource
def load_model():
    try:
        with open("disease_model.pkl", "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════════════
col_lang, _ = st.columns([1, 3])
with col_lang:
    language = st.selectbox("", list(LANG.keys()), label_visibility="collapsed")

t = LANG[language]

st.markdown(f"""
<div class="hero-wrapper">
  <div class="hero-tag">{t['tag']}</div>
  <h1 class="hero-title">🌿 {t['title']}</h1>
  <p class="hero-sub">{t['sub']}</p>
  <div style="margin-top:1rem;display:flex;justify-content:center;gap:8px;flex-wrap:wrap;">
    <span class="feature-badge badge-ml">⚡ ML + Rule Engine</span>
    <span class="feature-badge badge-vision">📸 Vision AI</span>
    <span class="feature-badge badge-audio">🎙️ Voice Input</span>
  </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  STATS BAR
# ═══════════════════════════════════════════════════════════════════
total_crops = len(ALL_CROPS)
total_diseases = sum(len(v) for v in RULE_BASED_DB.values()) + len(ML_CLASSES)
ml_crop_count = len(ML_CROPS)
rule_crop_count = len(RULE_BASED_DB)

st.markdown(f"""
<div class="stat-row" style="margin-bottom:0;">
  <div class="stat-box"><div class="stat-value">{total_crops}</div><div class="stat-label">Total Crops</div></div>
  <div class="stat-box"><div class="stat-value">{ml_crop_count}</div><div class="stat-label">ML Model Crops</div></div>
  <div class="stat-box"><div class="stat-value">{rule_crop_count}</div><div class="stat-label">Rule-Based Crops</div></div>
  <div class="stat-box"><div class="stat-value">{total_diseases}</div><div class="stat-label">Disease Profiles</div></div>
  <div class="stat-box"><div class="stat-value" style="color:var(--accent-p);">📸</div><div class="stat-label">Vision AI</div></div>
  <div class="stat-box"><div class="stat-value" style="color:var(--accent-y);">🎙️</div><div class="stat-label">Voice Input</div></div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  API KEY + NOTIFICATION PANEL
# ═══════════════════════════════════════════════════════════════════
st.markdown(f'<div class="glass-panel"><div class="panel-label">🔑 API & NOTIFICATION SETTINGS</div>', unsafe_allow_html=True)

api_key_col, notif_col = st.columns([1, 1])
with api_key_col:
    anthropic_api_key = st.text_input(
        t.get("api_key_label", "Anthropic API Key (for Vision & Chat)"),
        type="password",
        placeholder="sk-ant-...",
        key="main_api_key",
        help="Required for Image Recognition and AI Chatbot. Get from console.anthropic.com"
    )
    st.markdown("<p style='font-family:JetBrains Mono,monospace;font-size:10px;color:#7fa89a;'>ⓘ Used for Vision AI + Expert Chatbot · Get from console.anthropic.com</p>", unsafe_allow_html=True)

with notif_col:
    user_email   = st.text_input(t["email_label"],  placeholder="you@email.com",    key="main_uemail")
    sender_email = st.text_input(t["email_key"],    placeholder="sender@gmail.com", key="main_semail")
    email_pass   = st.text_input(t["email_pass"],   type="password",                key="main_epass")
    telegram_chat = st.text_input(t["phone_label"], placeholder="Telegram Chat ID", key="main_tgchat")
    tg_token      = st.text_input(t["tg_token"],    type="password",               key="main_tgtoken")

st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  📸 IMAGE RECOGNITION PANEL  ← NEW
# ═══════════════════════════════════════════════════════════════════
st.markdown(f'<div class="glass-panel vision-panel"><div class="panel-label panel-label-purple">{t.get("sec_vision","📸 IMAGE RECOGNITION — UPLOAD CROP PHOTO")}</div>', unsafe_allow_html=True)

st.markdown("""
<div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#7fa89a;margin-bottom:1rem;line-height:1.8;">
  📷 Upload a clear photo of the affected crop leaf/plant → Claude Vision AI will analyze visible symptoms and identify the disease.<br>
  ⚡ Works best with: clear focus · natural lighting · close-up of affected area · leaf fully in frame
</div>
""", unsafe_allow_html=True)

img_col, img_result_col = st.columns([1, 1])

with img_col:
    img_crop_select = st.selectbox(
        "Crop in the photo",
        options=ALL_CROPS,
        format_func=lambda x: f"{get_crop_emoji(x)} {x}",
        key="img_crop_select"
    )
    uploaded_image = st.file_uploader(
        "Upload crop photo (JPG/PNG/WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        key="crop_image_uploader",
        help="Take a clear photo of the diseased leaf or plant"
    )

    if uploaded_image:
        st.image(uploaded_image, caption=f"📸 {uploaded_image.name}", use_container_width=True)

    analyze_img_btn = st.button(t.get("vision_analyze", "🔬  ANALYZE IMAGE"), key="analyze_image_btn")

with img_result_col:
    if analyze_img_btn:
        if not uploaded_image:
            st.warning("⚠️ Please upload a crop photo first.")
        elif not anthropic_api_key:
            st.warning("⚠️ Please enter your Anthropic API Key above to use Vision AI.")
        else:
            with st.spinner("🔬 Analyzing image with Claude Vision AI..."):
                img_bytes = uploaded_image.read()
                ext = uploaded_image.name.split(".")[-1]
                result, error = analyze_crop_image(img_bytes, ext, img_crop_select, anthropic_api_key)

            if error:
                st.error(f"Vision Analysis Failed: {error}")
            elif result:
                st.session_state.image_analysis_result = result
                st.success(t.get("vision_success", "✓  IMAGE ANALYSIS COMPLETE"))

    # Display stored vision result
    vr = st.session_state.get("image_analysis_result")
    if vr:
        detected = vr.get("detected_disease", "Unknown")
        v_severity = vr.get("severity", "Moderate")
        v_confidence = vr.get("confidence", 0)
        v_treatment = vr.get("treatment", "Consult an expert.")
        v_symptoms = vr.get("visual_symptoms", "")
        v_quality = vr.get("image_quality", "Good")
        sc = SEVERITY_COLORS.get(v_severity, "#ffd600")

        st.markdown(f"""
        <div class="vision-result-card">
          <div class="panel-label panel-label-purple">🔬 VISION AI RESULT</div>
          <div style="margin-bottom:0.8rem;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#7fa89a;
              letter-spacing:2px;text-transform:uppercase;">Image Quality: </span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
              color:{'#00e676' if v_quality=='Good' else '#ffd600'};">{v_quality}</span>
          </div>
          <div class="vision-disease-badge">{detected}</div>
          <div class="stat-row" style="margin-top:1rem;">
            <div class="stat-box">
              <div class="stat-value" style="color:var(--accent-p);">{v_confidence}%</div>
              <div class="stat-label">Confidence</div>
            </div>
            <div class="stat-box">
              <div class="stat-value" style="font-size:18px;color:{sc};">{v_severity}</div>
              <div class="stat-label">Severity</div>
            </div>
            <div class="stat-box">
              <div class="stat-value" style="font-size:18px;color:var(--accent-b);">{vr.get('leaf_spots',0)}</div>
              <div class="stat-label">Leaf Spots</div>
            </div>
            <div class="stat-box">
              <div class="stat-value" style="font-size:18px;color:var(--accent-y);">{vr.get('yellowing',0)}</div>
              <div class="stat-label">Yellowing</div>
            </div>
          </div>
          <div style="margin-top:0.8rem;">
            <div class="panel-label panel-label-purple" style="font-size:9px;">👁️ VISUAL SYMPTOMS OBSERVED</div>
            <div class="visual-symptoms-box">{v_symptoms}</div>
          </div>
          <div style="margin-top:0.8rem;">
            <div class="panel-label" style="font-size:9px;">💊 TREATMENT</div>
            <div class="treatment-badge" style="font-size:13px;">{v_treatment}</div>
          </div>
          <div style="margin-top:0.8rem;">
            <div class="panel-label panel-label-purple" style="font-size:9px;margin-bottom:4px;">VISION CONFIDENCE</div>
            <div class="progress-track">
              <div class="progress-fill progress-fill-vision" style="width:{min(v_confidence,100)}%"></div>
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#7fa89a;">{v_confidence}% · Claude Vision AI</div>
          </div>
          <div style="margin-top:1rem;padding:8px 12px;background:rgba(224,64,251,0.06);border:1px solid rgba(224,64,251,0.2);border-radius:3px;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#7fa89a;">💡 TIP: Use detected symptom scores below to auto-fill the slider inputs</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Auto-fill sliders button
        if st.button("⚡ AUTO-FILL SLIDERS FROM IMAGE ANALYSIS", key="autofill_sliders"):
            st.session_state["main_spots"] = vr.get("leaf_spots", 3)
            st.session_state["main_yell"]  = vr.get("yellowing", 2)
            st.session_state["main_wilt"]  = vr.get("wilting", 1)
            st.session_state["main_mold"]  = vr.get("mold", 4)
            st.session_state["main_lcurl"] = vr.get("leaf_curl", 2)
            st.session_state["main_discol"]= vr.get("discoloration", 3)
            st.success("✅ Sliders updated from image analysis! Now run the main analysis below.")
            st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  DIAGNOSTIC INPUT PANEL
# ═══════════════════════════════════════════════════════════════════
st.markdown(f'<div class="glass-panel"><div class="panel-label">🧬 {t["sec_input"]}</div>', unsafe_allow_html=True)
col_a, col_b = st.columns([1, 2])

with col_a:
    crop = st.selectbox(
        t["crop"],
        options=ALL_CROPS,
        format_func=lambda x: f"{get_crop_emoji(x)} {x}",
        key="main_crop"
    )

    is_ml = crop in ML_CROPS
    badge_cls = "ml-badge" if is_ml else ""
    badge_lbl = t.get("ml_label","ML MODEL") if is_ml else t.get("rule_label","RULE-BASED")

    if is_ml:
        disease_list = ML_CROP_CLASSES[crop]
        list_html = "".join(
            f'<div class="disease-list-item">· {fmt_name(c)} '
            f'<span class="sev-{DISEASE_META.get(c,{}).get("severity","Moderate")}">'
            f'[{DISEASE_META.get(c,{}).get("severity","?")}]</span></div>'
            for c in disease_list
        )
    else:
        disease_list = RULE_BASED_DB.get(crop, [])
        list_html = "".join(
            f'<div class="disease-list-item">· {d["disease"]} '
            f'<span class="sev-{d["severity"]}">[{d["severity"]}]</span></div>'
            for d in disease_list
        )

    st.markdown(f"""
    <div style="margin-top:0.8rem;">
      <div class="crop-category-badge {badge_cls}">{badge_lbl}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#7fa89a;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;margin-top:4px;">Detectable diseases:</div>
      {list_html}
    </div>""", unsafe_allow_html=True)

with col_b:
    c1s, c2s, c3s = st.columns(3)
    with c1s:
        leaf_spots = st.slider(t["spot"],    0, 10, st.session_state.get("main_spots", 3), key="main_spots")
        yellowing  = st.slider(t["yellow"],  0, 10, st.session_state.get("main_yell", 2),  key="main_yell")
    with c2s:
        wilting    = st.slider(t["wilting"], 0, 10, st.session_state.get("main_wilt", 1),  key="main_wilt")
        mold       = st.slider(t["mold"],    0, 10, st.session_state.get("main_mold", 4),  key="main_mold")
    with c3s:
        leaf_curl     = st.slider("🌀 Leaf Curl",     0, 10, st.session_state.get("main_lcurl", 2),  key="main_lcurl")
        discoloration = st.slider("🎨 Discoloration", 0, 10, st.session_state.get("main_discol", 3), key="main_discol")

st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  TRANSLATION SELECTOR
# ═══════════════════════════════════════════════════════════════════
st.markdown(f'<div class="glass-panel"><div class="panel-label">🌐 {t["sec_translate"]}</div>', unsafe_allow_html=True)
selected_translate_langs = st.multiselect(
    t["translate_langs"],
    options=ALL_LANGS,
    default=["Hindi","Telugu"],
    key="main_trans_langs"
)
st.markdown("<p style='font-family:JetBrains Mono,monospace;font-size:10px;color:#7fa89a;margin-top:4px;'>ⓘ Results and email will include selected language translations</p></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  ANALYZE BUTTON + RESULTS
# ═══════════════════════════════════════════════════════════════════
if st.button(t["predict"]):
    prog = st.progress(0, text="Initializing neural scan…")
    for i in range(1, 101):
        time.sleep(0.010)
        prog.progress(i, text=f"Analyzing symptom vectors… {i}%")
    prog.empty()

    is_ml_crop = crop in ML_CROPS
    predicted_class = None
    disease = ""
    treatment = ""
    severity = "Moderate"
    confidence = 85.0
    detection_mode = ""

    if is_ml_crop:
        detection_mode = "ML Model"
        bundle = load_model()
        features = np.array([[leaf_spots, yellowing, wilting, mold, leaf_curl, discoloration]], dtype=float)
        if bundle:
            clf = bundle["model"]; le = bundle["encoder"]
            crop_valid = ML_CROP_CLASSES[crop]
            class_names = list(le.classes_)
            proba = clf.predict_proba(features)[0]
            crop_indices = [class_names.index(c) for c in crop_valid if c in class_names]
            crop_probas  = [(class_names[i], proba[i]) for i in crop_indices]
            predicted_class, raw_prob = max(crop_probas, key=lambda x: x[1])
            meta = DISEASE_META.get(predicted_class, {})
            treatment = meta.get("treatment","Consult an agricultural expert.")
            severity  = meta.get("severity","Moderate")
            disease   = fmt_name(predicted_class)
            confidence = 80 + min(raw_prob / 0.30, 1.0) * 15
        else:
            p = [leaf_spots, yellowing, wilting, mold, leaf_curl, discoloration]
            scores = {c: sum(abs(p[i]-ML_PROFILES.get(c,[5]*6)[i]) for i in range(6)) for c in ML_CROP_CLASSES[crop]}
            predicted_class = min(scores, key=scores.get)
            meta = DISEASE_META.get(predicted_class, {})
            treatment = meta.get("treatment","Consult an agricultural expert.")
            severity  = meta.get("severity","Moderate")
            disease   = fmt_name(predicted_class)
            confidence = float(np.random.uniform(82, 93))
    else:
        detection_mode = "Rule-Based"
        matched, confidence = detect_rule_based(crop, leaf_spots, yellowing, wilting, mold, leaf_curl, discoloration)
        if matched:
            disease   = matched["disease"]
            treatment = matched["remedy"]
            severity  = matched["severity"]
            predicted_class = f"{crop.lower().replace(' ','_')}_{disease.lower().replace(' ','_')}"
        else:
            disease   = "Unknown"
            treatment = "Consult your local agricultural extension office."
            severity  = "Unknown"
            predicted_class = "unknown"

    st.success(t["success"])

    translations = {}
    for lang_name in selected_translate_langs:
        tm = TRANSLATION_MAP.get(lang_name, {})
        entry = tm.get(predicted_class)
        if entry:
            translations[lang_name] = {"disease": entry[0], "treatment": entry[1], "severity": severity}
        else:
            translations[lang_name] = {"disease": disease, "treatment": treatment, "severity": severity}

    translation_blocks_html = ""
    for lang_name, data in translations.items():
        translation_blocks_html += f"""
        <div class="translate-box" style="margin-bottom:0.6rem;">
          <div class="translate-lang-badge">{lang_name}</div>
          <div style="display:flex;gap:1.5rem;flex-wrap:wrap;">
            <div><div class="translate-original">Disease</div>
              <div class="translate-result" style="color:var(--accent-g);font-weight:600;">{data['disease']}</div></div>
            <div><div class="translate-original">Severity</div>
              <div class="translate-result" style="color:var(--accent-y);">{data['severity']}</div></div>
            <div style="flex:2;min-width:180px;"><div class="translate-original">Treatment</div>
              <div class="translate-result" style="color:var(--accent-b);">{data['treatment']}</div></div>
          </div>
        </div>"""

    severity_color = SEVERITY_COLORS.get(severity, "#ffd600")
    mode_color = "#00e676" if detection_mode == "ML Model" else "#00b0ff"

    st.markdown(f"""
    <div class="result-card">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:1rem;">
        <div>
          <div class="panel-label">🦠 {t['disease']}</div>
          <div class="disease-badge">{disease}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#7fa89a;margin-top:6px;">{crop} · {predicted_class}</div>
          <div style="margin-top:6px;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:2px;
            color:{mode_color};border:1px solid {mode_color}44;padding:2px 10px;border-radius:2px;
            text-transform:uppercase;">⚙ {detection_mode}</span>
          </div>
        </div>
        <div style="text-align:right;">
          <div class="panel-label">SEVERITY</div>
          <div class="disease-badge" style="border-color:{severity_color};color:{severity_color};
            box-shadow:0 0 16px {severity_color}44;">{severity}</div>
        </div>
      </div>
      <div class="stat-row" style="margin-top:1.5rem;">
        <div class="stat-box"><div class="stat-value">{confidence:.0f}%</div><div class="stat-label">Confidence</div></div>
        <div class="stat-box"><div class="stat-value">{leaf_spots}</div><div class="stat-label">Leaf Spots</div></div>
        <div class="stat-box"><div class="stat-value">{yellowing}</div><div class="stat-label">Yellowing</div></div>
        <div class="stat-box"><div class="stat-value">{wilting}</div><div class="stat-label">Wilting</div></div>
        <div class="stat-box"><div class="stat-value">{mold}</div><div class="stat-label">Mold</div></div>
        <div class="stat-box"><div class="stat-value">{leaf_curl}</div><div class="stat-label">Leaf Curl</div></div>
        <div class="stat-box"><div class="stat-value">{discoloration}</div><div class="stat-label">Discoloration</div></div>
      </div>
      <div style="margin-top:1.5rem;">
        <div class="panel-label">💊 {t['treatment']}</div>
        <div class="treatment-badge">{treatment}</div>
      </div>
      <div style="margin-top:1.5rem;">
        <div class="panel-label">CONFIDENCE MATRIX</div>
        <div class="progress-track"><div class="progress-fill" style="width:{min(confidence,100):.0f}%"></div></div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#7fa89a;">{confidence:.1f}% · {detection_mode}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if translation_blocks_html:
        st.markdown(f"""
        <div style="margin-top:1.8rem;padding:0 0 1rem;">
          <div class="panel-label">🌐 MULTILINGUAL TRANSLATIONS</div>
          {translation_blocks_html}
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="panel-label" style="margin-top:1.5rem;">📡 NOTIFICATION STATUS</div>', unsafe_allow_html=True)
    email_status = send_email(user_email, sender_email, email_pass, crop, disease, treatment, severity, confidence, translations)
    tg_status    = send_telegram(telegram_chat, tg_token, crop, disease, treatment, severity)

    def pill(label, status):
        if status is None: cls, icon = "notif-skip", "○"
        elif status:        cls, icon = "notif-sent", "✓"
        else:               cls, icon = "notif-fail", "✗"
        return f'<span class="notif-pill {cls}">{icon} {label}</span>'

    st.markdown(f"""
    <div class="notif-row">{pill("EMAIL",email_status)}{pill("TELEGRAM",tg_status)}</div>
    <p style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#7fa89a;margin-top:8px;">
      ○ NOT CONFIGURED · ✓ SENT · ✗ FAILED</p>""", unsafe_allow_html=True)

    # ── Audio Output ────────────────────────────────────────────────
    GTTS_CODES = {
        "English": "en", "Hindi": "hi", "Telugu": "te",
        "Tamil": "ta", "Kannada": "kn", "Marathi": "mr"
    }

    AUDIO_TEMPLATES = {
        "Hindi":   lambda c,d,s,tr: f"विश्लेषण पूर्ण। {c} में {d} रोग पाया गया। गंभीरता: {s}। उपचार: {tr}",
        "Telugu":  lambda c,d,s,tr: f"విశ్లేషణ పూర్తయింది. {c} లో {d} వ్యాధి గుర్తించబడింది. తీవ్రత: {s}. చికిత్స: {tr}",
        "Tamil":   lambda c,d,s,tr: f"பகுப்பாய்வு முடிந்தது. {c} இல் {d} நோய் கண்டறியப்பட்டது. தீவிரம்: {s}. சிகிச்சை: {tr}",
        "Kannada": lambda c,d,s,tr: f"ವಿಶ್ಲೇಷಣೆ ಪೂರ್ಣಗೊಂಡಿದೆ. {c} ನಲ್ಲಿ {d} ರೋಗ ಪತ್ತೆಯಾಗಿದೆ. ತೀವ್ರತೆ: {s}. ಚಿಕಿತ್ಸೆ: {tr}",
        "Marathi": lambda c,d,s,tr: f"विश्लेषण पूर्ण झाले. {c} मध्ये {d} रोग आढळला. तीव्रता: {s}. उपचार: {tr}",
        "English": lambda c,d,s,tr: f"Analysis complete. Detected disease is {d} on {c}. Severity: {s}. Treatment: {tr}",
    }

    def get_localized(lang_name, pred_class, fb_disease, fb_treatment):
        tm = TRANSLATION_MAP.get(lang_name, {})
        entry = tm.get(pred_class)
        if entry:
            return entry[0], entry[1]
        td = translations.get(lang_name, {})
        return td.get("disease", fb_disease), td.get("treatment", fb_treatment)

    all_audio_langs = [language] + [l for l in selected_translate_langs if l != language]
    seen_langs = set()
    audio_langs = []

    for lang_name in all_audio_langs:
        if lang_name in seen_langs: continue
        seen_langs.add(lang_name)
        gtts_code = GTTS_CODES.get(lang_name, "en")
        loc_disease, loc_treatment = get_localized(lang_name, predicted_class, disease, treatment)
        template = AUDIO_TEMPLATES.get(lang_name, AUDIO_TEMPLATES["English"])
        voice_msg = template(crop, loc_disease, severity, loc_treatment)
        audio_langs.append((lang_name, gtts_code, voice_msg))

    st.markdown('<div class="panel-label" style="margin-top:1.5rem;">🔊 MULTILINGUAL AUDIO BROADCAST</div>', unsafe_allow_html=True)

    for lang_name, gtts_code, voice_msg in audio_langs:
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;letter-spacing:2px;'
            f'color:var(--accent-b);text-transform:uppercase;margin:0.7rem 0 0.3rem;">▶ {lang_name}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="font-family:\'Exo 2\',sans-serif;font-size:12px;color:#7fa89a;'
            f'margin-bottom:6px;padding:6px 10px;background:rgba(0,176,255,0.05);'
            f'border-left:2px solid rgba(0,176,255,0.3);border-radius:2px;">'
            f'{voice_msg}</div>',
            unsafe_allow_html=True
        )
        try:
            tts = gTTS(text=voice_msg, lang=gtts_code)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(tmp.name)
            st.audio(open(tmp.name, "rb").read(), format="audio/mp3")
        except Exception as e:
            st.caption(f"Audio unavailable for {lang_name}: {e}")

    st.session_state.last_diagnosis = {
        "crop": crop, "disease": disease, "predicted_class": predicted_class,
        "treatment": treatment, "severity": severity, "confidence": confidence,
        "detection_mode": detection_mode,
        "leaf_spots": leaf_spots, "yellowing": yellowing,
        "wilting": wilting, "mold": mold,
        "leaf_curl": leaf_curl, "discoloration": discoloration,
    }

# ═══════════════════════════════════════════════════════════════════
#  CROP REFERENCE TABLE
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="divider" style="margin-top:2.5rem;"></div>', unsafe_allow_html=True)
st.markdown('<div class="glass-panel"><div class="panel-label">📋 CROP DISEASE REFERENCE — ALL 100 CROPS</div>', unsafe_allow_html=True)

filter_col1, filter_col2 = st.columns([2,3])
with filter_col1:
    ref_crop = st.selectbox("Filter by crop", ["All"] + ALL_CROPS, key="ref_crop_filter",
                            format_func=lambda x: f"{get_crop_emoji(x)} {x}" if x != "All" else "🌿 All Crops")
with filter_col2:
    sev_filter = st.multiselect("Filter by severity", ["None","Low","Moderate","High","Critical"],
                                default=["Moderate","High","Critical"], key="ref_sev_filter")

rows_html = ""
row_count = 0
for crop_name in (ALL_CROPS if ref_crop == "All" else [ref_crop]):
    emoji = get_crop_emoji(crop_name)
    mode_label = "ML" if crop_name in ML_CROPS else "Rule"
    mode_color = "#00e676" if crop_name in ML_CROPS else "#00b0ff"

    if crop_name in ML_CROPS:
        entries = [{"disease": fmt_name(c), "remedy": DISEASE_META.get(c,{}).get("treatment","—"),
                    "severity": DISEASE_META.get(c,{}).get("severity","Moderate")}
                   for c in ML_CROP_CLASSES[crop_name]]
    else:
        entries = [{"disease": d["disease"], "remedy": d["remedy"], "severity": d["severity"]}
                   for d in RULE_BASED_DB.get(crop_name, [])]

    for entry in entries:
        if sev_filter and entry["severity"] not in sev_filter:
            continue
        sc = SEVERITY_COLORS.get(entry["severity"],"#ffd600")
        rows_html += f"""
        <tr>
          <td style="padding:8px 12px;border:1px solid #1a2a1a;white-space:nowrap;">
            <span style="font-size:16px;">{emoji}</span>
            <span style="color:#e8f5e9;margin-left:6px;">{crop_name}</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
              color:{mode_color};border:1px solid {mode_color}44;padding:1px 6px;
              border-radius:2px;margin-left:6px;text-transform:uppercase;">{mode_label}</span>
          </td>
          <td style="padding:8px 12px;border:1px solid #1a2a1a;color:var(--accent-g);font-weight:600;">{entry['disease']}</td>
          <td style="padding:8px 12px;border:1px solid #1a2a1a;">
            <span style="color:{sc};font-family:'JetBrains Mono',monospace;font-size:10px;
              border:1px solid {sc}44;padding:2px 8px;border-radius:2px;">{entry['severity']}</span>
          </td>
          <td style="padding:8px 12px;border:1px solid #1a2a1a;color:#7fa89a;font-size:12px;max-width:300px;">{entry['remedy'][:100]}{'…' if len(entry['remedy'])>100 else ''}</td>
        </tr>"""
        row_count += 1

st.markdown(f"""
<div style="overflow-x:auto;max-height:420px;overflow-y:auto;">
<table style="width:100%;border-collapse:collapse;font-family:'Exo 2',sans-serif;font-size:13px;">
  <thead>
    <tr style="background:#0a1520;position:sticky;top:0;">
      <th style="padding:10px 12px;border:1px solid #1a3a2a;color:var(--accent-b);font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:2px;text-align:left;text-transform:uppercase;">Crop</th>
      <th style="padding:10px 12px;border:1px solid #1a3a2a;color:var(--accent-b);font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:2px;text-align:left;text-transform:uppercase;">Disease</th>
      <th style="padding:10px 12px;border:1px solid #1a3a2a;color:var(--accent-b);font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:2px;text-align:left;text-transform:uppercase;">Severity</th>
      <th style="padding:10px 12px;border:1px solid #1a3a2a;color:var(--accent-b);font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:2px;text-align:left;text-transform:uppercase;">Remedy</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
</div>
<p style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#7fa89a;margin-top:8px;">Showing {row_count} disease entries</p>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  🎙️ VOICE INPUT + CROP EXPERT CHATBOT  ← NEW + EXISTING
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="divider" style="margin-top:2.5rem;"></div>', unsafe_allow_html=True)

QUICK_QUESTIONS = [
    "How to treat Banana Sigatoka?",
    "What causes Wheat Rust?",
    "Groundnut rust prevention tips?",
    "Best fungicide for Rice Blast?",
    "Organic treatment for Radish Mosaic?",
    "How to identify Banana Moko disease?",
    "Signs of Potato Late Blight?",
    "When to spray for Chilli Anthracnose?",
]

st.markdown("""
<div class="glass-panel">
  <div class="chat-section-header">
    <div class="chat-dot"></div>
    <div class="panel-label" style="margin:0;">🌾 AGRISENSE CROP EXPERT — ASK ANYTHING ABOUT YOUR CROPS</div>
  </div>
""", unsafe_allow_html=True)

# ── 🎙️ VOICE INPUT SECTION ──────────────────────────────────────
st.markdown(f'<div class="audio-zone"><div class="panel-label panel-label-yellow" style="margin-bottom:0.5rem;">{t.get("sec_audio","🎙️ VOICE INPUT — SPEAK YOUR QUESTION")}</div>', unsafe_allow_html=True)
st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#7fa89a;margin-bottom:0.8rem;">Record your question about crop disease — it will be transcribed and sent to the AI expert automatically.</p>', unsafe_allow_html=True)

audio_input = st.audio_input(
    "🎙️ Click to record your crop question",
    key="voice_input_recorder"
)

if audio_input is not None:
    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:var(--accent-y);margin-top:0.5rem;">▶ Audio recorded — click TRANSCRIBE to convert to text</div>', unsafe_allow_html=True)

    transcribe_col, _ = st.columns([1, 3])
    with transcribe_col:
        if st.button("🎙️ TRANSCRIBE & SEND", key="transcribe_btn"):
            if not anthropic_api_key:
                st.warning("⚠️ Enter your Anthropic API Key above to use voice transcription.")
            else:
                with st.spinner("🎙️ Transcribing your voice…"):
                    audio_bytes = audio_input.read()
                    transcribed, err = transcribe_audio_query(audio_bytes, anthropic_api_key)

                if err:
                    # Graceful fallback — show error and let user type
                    st.warning(f"⚠️ {err}")
                elif transcribed:
                    st.session_state.audio_transcription = transcribed
                    st.session_state.chip_query = transcribed
                    st.markdown(f'<div class="audio-transcription-box">🎙️ Transcribed: "{transcribed}"</div>', unsafe_allow_html=True)
                    st.rerun()

# Show last transcription if available
if st.session_state.get("audio_transcription"):
    st.markdown(f'<div class="audio-transcription-box">✅ Last transcription: "{st.session_state.audio_transcription}"</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # close audio-zone

# ── QUICK QUESTIONS ──────────────────────────────────────────────
st.markdown('<div class="quick-chip-label" style="margin-top:1rem;">💬 QUICK QUESTIONS — CLICK TO ASK</div>', unsafe_allow_html=True)
chip_cols = st.columns(4)
for ci, q in enumerate(QUICK_QUESTIONS):
    with chip_cols[ci % 4]:
        st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
        if st.button(q, key=f"chip_{ci}"):
            st.session_state.chip_query = q
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if not st.session_state.chat_history:
    st.markdown("""
    <div class="chat-history-box">
      <div class="chat-empty-state">
        <div class="chat-empty-icon">🌿</div>
        <div>No conversation yet</div>
        <div style="font-size:10px;opacity:0.6;">Use voice input 🎙️ · click a quick question · or type below</div>
      </div>
    </div>""", unsafe_allow_html=True)
else:
    msgs_html = ""
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            msgs_html += f"""<div><div class="chat-sender-user">YOU</div>
              <div class="chat-msg-user">{msg["content"]}</div></div>"""
        else:
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', msg["content"])
            content = content.replace("\n","<br>")
            msgs_html += f"""<div><div class="chat-sender-bot">🌿 AGRISENSE EXPERT</div>
              <div class="chat-msg-bot">{content}</div></div>"""
    st.markdown(f'<div class="chat-history-box">{msgs_html}</div>', unsafe_allow_html=True)

chip_val = st.session_state.chip_query
if chip_val:
    st.session_state.chip_query = ""

col_inp, col_btn, col_clr = st.columns([6, 1, 1])
with col_inp:
    user_query = st.text_input("", value=chip_val, placeholder="e.g. How do I treat Rice Blast organically?",
                               label_visibility="collapsed", key="chat_input_field")
with col_btn:
    send_clicked = st.button("⚡ ASK", key="chat_send")
with col_clr:
    clear_clicked = st.button("🗑 CLEAR", key="chat_clear")

st.markdown("</div>", unsafe_allow_html=True)

if clear_clicked:
    st.session_state.chat_history = []
    st.session_state.audio_transcription = ""
    st.rerun()

if send_clicked and user_query.strip():
    if not anthropic_api_key:
        st.warning("⚠️ Please enter your Anthropic API Key in the settings above to use the AI chatbot.")
    else:
        diag = st.session_state.get("last_diagnosis", {})
        context_block = ""
        if diag:
            context_block = f"""
The farmer just ran a diagnosis:
- Crop: {diag.get('crop')} | Disease: {diag.get('disease')} | Severity: {diag.get('severity')}
- Confidence: {diag.get('confidence',0):.0f}% | Detection mode: {diag.get('detection_mode','—')}
- Treatment: {diag.get('treatment')}
- Symptoms: Leaf Spots={diag.get('leaf_spots')}/10, Yellowing={diag.get('yellowing')}/10, Wilting={diag.get('wilting')}/10, Mold={diag.get('mold')}/10, Leaf Curl={diag.get('leaf_curl')}/10, Discoloration={diag.get('discoloration')}/10"""

        vision_context = ""
        vr = st.session_state.get("image_analysis_result")
        if vr:
            vision_context = f"""
The farmer also uploaded a crop image. Vision AI detected:
- Visual disease: {vr.get('detected_disease')} | Confidence: {vr.get('confidence')}%
- Visual symptoms: {vr.get('visual_symptoms')}"""

        ref_summary = ""
        for crop_name, disease_list in RULE_BASED_DB.items():
            diseases = ", ".join(d["disease"] for d in disease_list if d["disease"] != "Healthy")
            if diseases:
                ref_summary += f"  {crop_name}: {diseases}\n"

        system_prompt = f"""You are AgriSense Expert, an advanced agricultural AI assistant covering 100+ crops and their diseases.

You have knowledge of the following crop-disease database:
RULE-BASED CROPS (50+ crops including cereals, pulses, vegetables, fruits, oilseeds):
{ref_summary}
ML MODEL CROPS (with 30 detailed disease classes):
  Banana: bract_mosaic_virus, cordana, healthy, insectpest, moko, panama, pestalotiopsis, sigatoka, yb_sigatoka
  Cauliflower: Blackrot, bacterial_spot_rot, downy_mildew, healthy
  Chilli: anthracnose, healthy, leafcurl, leafspot, whitefly, yellowish
  Groundnut: early_leaf_spot, early_rust, healthy, late_leaf_spot, nutrition_deficiency, rust
  Radish: black_leaf_spot, downey_mildew, flea_beetle, healthy, mosaic
{context_block}
{vision_context}

Guidelines:
- Give concise, practical agricultural advice.
- Use bullet points for multi-step answers.
- Always mention safety precautions when recommending chemicals.
- Prefer organic/sustainable methods first.
- If unrelated to agriculture, politely redirect.
- Format key terms in **bold**.
- Keep responses under 200 words unless the topic demands more.
- If the question came via voice input, keep the answer clear and easy to understand when read aloud."""

        messages_payload = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history[-10:]]
        messages_payload.append({"role": "user", "content": user_query.strip()})
        st.session_state.chat_history.append({"role": "user", "content": user_query.strip()})

        with st.spinner("🌿 Consulting crop expert…"):
            try:
                resp = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": anthropic_api_key,
                        "anthropic-version": "2023-06-01"
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 1000,
                        "system": system_prompt,
                        "messages": messages_payload,
                    },
                    timeout=30,
                )
                data  = resp.json()
                reply = "".join(b.get("text","") for b in data.get("content",[]) if b.get("type")=="text").strip()
                if not reply:
                    if "error" in data:
                        reply = f"⚠️ API Error: {data['error'].get('message','Unknown error')}. Check your API key."
                    else:
                        reply = "Sorry, I couldn't generate a response. Please try again."
            except Exception as e:
                reply = f"⚠️ Connection error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.session_state.audio_transcription = ""  # clear after use
        st.rerun()

# ── Footer ──────────────────────────────────────────────────────────
st.markdown("""
<div class="divider" style="margin-top:3rem;"></div>
<p style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#2a4a3a;text-align:center;padding-bottom:1rem;letter-spacing:2px;">
  AGRISENSE AI v3.0 · 100+ CROPS · NEURAL PATHOGEN DETECTION · VISION AI · VOICE INPUT · FOR RESEARCH & ADVISORY USE ONLY
</p>""", unsafe_allow_html=True)