# 1. Initialize the database
# Start-Process powershell -NoNewWindow -ArgumentList "cd ./detection_service; del violations.db"
Start-Process powershell -NoNewWindow -ArgumentList "python init_db_script.py"
# Start-Sleep -Seconds 5  # <- Add this delay

# PowerShell Script to Run All 3 Services with multi-processing

# Set environment variable to enable multi-processing in Python
# $env:PYTHONUNBUFFERED = "1"
# $env:PIZZA_MONITORING_WORKERS = "4"  # Set number of worker processes

# Start each service in its own process
Start-Process powershell -NoNewWindow -ArgumentList "cd ./frame_reader; python reader.py"
Start-Process powershell -NoNewWindow -ArgumentList "cd ./detection_service; python detector.py"
Start-Process powershell -NoNewWindow -ArgumentList "cd ./streaming_service; python stream_api.py"
