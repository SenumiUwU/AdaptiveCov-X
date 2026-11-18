@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo AdaptiveCov-X ^| Yahoo Finance data ^| 2014-01-01 to 2026-11-11
echo ============================================================
echo Installing Python dependencies (internet required)...
py -m pip install -r requirements.txt
if errorlevel 1 goto fail
echo.
echo Downloading Yahoo historical data and running backtest...
py run_poc.py --yahoo
if errorlevel 1 goto fail
echo.
echo PASSED. Opening real-price dashboard...
start "" "results_yahoo\viva_dashboard.html"
echo Data CSV: data\real_prices.csv
echo Audit:    data\yahoo_download_metadata.json
echo Results:  results_yahoo\test_metrics.csv
pause
exit /b 0
:fail
echo.
echo FAILURE. Yahoo Finance may be unavailable, rate-limited, or the Python setup incomplete.
echo No synthetic data has been substituted for actual Yahoo prices.
pause
exit /b 1
