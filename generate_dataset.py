import numpy as np
import csv, random, pickle

random.seed(42)
np.random.seed(42)

CLASSES = [
    'banana_bract_mosaic_virus','banana_cordana','banana_healthy','banana_insectpest',
    'banana_moko','banana_panama','banana_pestalotiopsis','banana_sigatoka','banana_yb_sigatoka',
    'cauliflower_Blackrot','cauliflower_bacterial_spot_rot','cauliflower_downy_mildew','cauliflower_healthy',
    'chilli_anthracnose','chilli_healthy','chilli_leafcurl','chilli_leafspot','chilli_whitefly',
    'chilli_yellowish','groundnut_early_leaf_spot','groundnut_early_rust','groundnut_healthy',
    'groundnut_late_leaf_spot','groundnut_nutrition_deficiency','groundnut_rust',
    'radish_black_leaf_spot','radish_downey_mildew','radish_flea_beetle','radish_healthy','radish_mosaic'
]

PROFILES = {
    'banana_bract_mosaic_virus':      [2,5,2,1,6,7],
    'banana_cordana':                  [7,3,2,5,1,4],
    'banana_healthy':                  [0,0,0,0,0,0],
    'banana_insectpest':               [4,2,4,2,7,3],
    'banana_moko':                     [2,7,8,3,2,5],
    'banana_panama':                   [1,6,9,2,1,6],
    'banana_pestalotiopsis':           [8,4,3,6,2,5],
    'banana_sigatoka':                 [9,4,2,7,1,6],
    'banana_yb_sigatoka':              [8,5,3,6,2,7],
    'cauliflower_Blackrot':            [3,5,4,2,1,8],
    'cauliflower_bacterial_spot_rot':  [5,3,5,4,2,6],
    'cauliflower_downy_mildew':        [4,3,3,8,2,4],
    'cauliflower_healthy':             [0,0,0,0,0,0],
    'chilli_anthracnose':              [7,3,3,5,2,6],
    'chilli_healthy':                  [0,0,0,0,0,0],
    'chilli_leafcurl':                 [2,2,3,1,9,4],
    'chilli_leafspot':                 [8,3,2,4,3,5],
    'chilli_whitefly':                 [3,4,3,2,5,3],
    'chilli_yellowish':                [2,9,3,2,3,7],
    'groundnut_early_leaf_spot':       [6,3,2,4,1,4],
    'groundnut_early_rust':            [5,2,2,3,1,6],
    'groundnut_healthy':               [0,0,0,0,0,0],
    'groundnut_late_leaf_spot':        [8,4,3,6,2,5],
    'groundnut_nutrition_deficiency':  [2,7,4,1,3,7],
    'groundnut_rust':                  [6,3,3,5,2,7],
    'radish_black_leaf_spot':          [8,3,2,5,2,5],
    'radish_downey_mildew':            [4,3,3,8,2,4],
    'radish_flea_beetle':              [5,2,3,2,4,3],
    'radish_healthy':                  [0,0,0,0,0,0],
    'radish_mosaic':                   [3,5,3,2,6,6],
}

rows = []
for cls in CLASSES:
    profile = PROFILES[cls]
    n = 150 if 'healthy' in cls else 200
    for _ in range(n):
        if 'healthy' in cls:
            features = [max(0.0, min(10.0, float(np.random.normal(p, 0.5)))) for p in profile]
        else:
            features = [max(0.0, min(10.0, float(np.random.normal(p, 1.2)))) for p in profile]
        rows.append(features + [cls])

random.shuffle(rows)

with open('crop_disease_dataset.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['leaf_spots','yellowing','wilting','mold','leaf_curl','discoloration','label'])
    writer.writerows(rows)

print(f"Generated {len(rows)} rows across {len(CLASSES)} classes")