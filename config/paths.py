from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "Config"
OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_DIR = BASE_DIR / "cache"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [CONFIG_DIR, OUTPUT_DIR, CACHE_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Paths dictionary
PATHS = {
    'YOUTUBE': {
        'CONFIG': CONFIG_DIR / 'youtube',
        'OUTPUTS': OUTPUT_DIR / 'youtube',
        'CACHE': CACHE_DIR / 'youtube',
        'LOGS': LOGS_DIR / 'youtube'
    }
}

# Create YouTube-specific directories
for path in PATHS['YOUTUBE'].values():
    path.mkdir(parents=True, exist_ok=True) 