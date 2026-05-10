@echo off
REM Limpia caché de MIOpen
setlocal enabledelayedexpansion

del /s /q "%APPDATA%\.miopen" 2>nul
del /s /q "%TEMP%\comgr*" 2>nul

REM Variables más agresivas
set MIOPEN_DISABLE_CACHE=1
set MIOPEN_FIND_MODE=NORMAL
set MIOPEN_CUSTOM_CACHE_DIR=%TEMP%\miopen_cache
set MIOPEN_COMPILE_PARALLEL_LEVEL=1
set MIOPEN_FORCE_KERNEL_LDS=0
set PYTORCH_ALLOC_CONF=expandable_segments:True
set HSA_OVERRIDE_GFX_VERSION=11.0.0
set MIOPEN_DEBUG_DISABLE_DROPOUT = 1

python train_crnn.py --no-amp --epochs 30 --batch-size 32