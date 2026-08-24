"""
Parâmetros globais do projeto COLOVO.
"""

# =========================
# Image
# =========================

IMAGE_SIZE = 256
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# =========================
# Classical segmentation
# =========================

# HSV threshold
LOWER_HSV = (8, 120, 60)
UPPER_HSV = (25, 255, 255)

# Component selection
MIN_COMPONENT_AREA = 1000
MIN_COMPONENT_CIRCULARITY = 0.15
MIN_COMPONENT_SOLIDITY = 0.70

# Expected yolk color
YOLK_HUE_MIN = 8
YOLK_HUE_MAX = 25
YOLK_MIN_SATURATION = 120

# Morphological cleanup
MORPH_KERNEL = 5
MORPH_OPEN_ITERATIONS = 1
MORPH_CLOSE_ITERATIONS = 2

# Segmentation decision
MIN_MASK_PIXELS = 100
SEGMENTATION_THRESHOLD = 0.5
FORCE_UNET = False

# =========================
# Segmentation model
# =========================

INNER_EROSION_KERNEL = 9
INNER_EROSION_ITERATIONS = 2

# =========================
# Feature extraction
# =========================

MIN_VALID_PIXELS = 50
MIN_SATURATION = 50
MAX_VALUE = 240

# =========================
# Random Forest
# =========================

RF_ESTIMATORS = 100
RF_RANDOM_STATE = 42

DSM_MIN = 1
DSM_MAX = 16

# =========================
# Training
# =========================

TRAIN_BATCH_SIZE = 8
TRAIN_EPOCHS = 40
TRAIN_LR = 1e-3

TRAIN_RATIO = 0.80
RANDOM_SEED = 42
