# 🍕 Pizza Store Hygiene Monitoring System

A real-time computer vision system that monitors hygiene protocol compliance in pizza preparation areas by detecting whether workers use scoopers when handling ingredients in designated regions of interest (ROIs).

## 🎯 Project Objectives

- Monitor food safety compliance in real-time
- Detect violations when workers handle ingredients without proper tools
- Provide visual alerts and statistics on compliance
- Create a scalable system that can handle multiple workers simultaneously

## 🧱 Microservices Architecture

This project uses a microservices architecture to ensure scalability, maintainability, and separation of concerns:

![System Diagram](https://github.com/amira-medhat/Pizza-Store-Hygiene-Monitoring-System/blob/main/assests/images/Block_Diagram.png?raw=true)

### 🛠 Services

1. **Frame Reader Service**: 
   - Captures video frames from cameras using OpenCV
   - Publishes frames to the message queue
   - Handles video input from files or live camera feeds

2. **Detection Service**: 
   - Processes frames using YOLOv12 object detection
   - Detects hands, scoopers, pizzas, and workers
   - Tracks objects across frames using ByteTrack
   - Stateless design for improved performance

3. **Violation Service**: 
   - Monitors designated ROIs (ingredient containers)
   - Detects when hands enter ROIs without scoopers
   - Records violations and safe pickups in the database
   - Maintains worker statistics

4. **Streaming Service**: 
   - Provides a web dashboard for monitoring
   - Displays real-time video with detection overlays
   - Shows violation statistics and alerts
   - Uses Flask and Waitress for robust web serving

### Communication

- **RabbitMQ**: Message broker for inter-service communication
- **SQLite**: Lightweight database for violation records
- **Flask with waitress server**: Real-time updates to the frontend

## 📦 System Requirements

- **GPU**: NVIDIA GTX 1660 Ti (or equivalent)
- **CPU**: Intel Core i7 (or equivalent)
- **RAM**: 16GB minimum
- **VRAM**: 6GB minimum
- **OS**: Windows 11
- **Python**: 3.9
- **CUDA**: 11.7 (for GPU acceleration)

## Setup Instructions

### 1. Environment Setup

```powershell
# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. RabbitMQ Setup

1. Install RabbitMQ Server from [rabbitmq.com](https://www.rabbitmq.com/download.html)
2. Start the RabbitMQ service:
   ```powershell
   net start RabbitMQ
   ```
3. Create the required queues:
   ```powershell
   # Install RabbitMQ management plugin if not already installed
   rabbitmq-plugins enable rabbitmq_management

   # Create queues (can be done through the management UI at http://localhost:15672)
   # Default credentials: guest/guest
   ```

### 3. Configure Paths

Edit the following files to match your system paths:

1. `config.py`: Update paths for video sources, database, and RabbitMQ settings
   ```python
   # Example configuration
   VIDEO_SOURCE = r"D:\PizzaStore_Task\your_video_file.mp4"
   DB_PATH = r"D:\PizzaStore_Task\pizza_monitoring\detection_service\violations.db"
   ```

2. `detection_logic.py`: Update model path
   ```python
   MODEL_PATH = r"D:\PizzaStore_Task\best.pt"
   ```

### 4. Initialize Database

Run the database initialization script:
```powershell
python init_db_script.py
```

### 5. Run the System

Start all services using the provided PowerShell script:
```powershell
.\run_all.ps1
```

This will launch:
- Frame Reader Service
- Detection Service
- Streaming Service

### 6. Access the Dashboard

Open your browser and navigate to:
```
http://localhost:8000
```

## 📽️ Demo

### Detection Logic Demo
- This video showcases how the detector service processes incoming frames, identifies worker violations or safe pickups.
![Detection Demo Thumbnail](https://github.com/amira-medhat/Pizza-Store-Hygiene-Monitoring-System/blob/main/assests/images/detection_thumbnail.png?raw=true)
🔗 Watch the Video: https://drive.google.com/file/d/1SaJffgct9BFOXKVHzT33Vpb-1Llt_U8y/view?usp=sharing

### Streaming Service Demo
- This video illustrates how the annotated frames (with bounding boxes, worker IDs, and violation alerts) are streamed live through the UI via the Flask-based streamer microservice.
![Streaming Service Demo Thumbnail](https://github.com/amira-medhat/Pizza-Store-Hygiene-Monitoring-System/blob/main/assests/images/Streaming_thumbnail.png?raw=true)
🔗 Watch the Video: https://drive.google.com/file/d/1iPb_gJYm-Ptjc0qTkFs2CS_76jdcXqF3/view?usp=sharing

## Performance Optimization

The system includes several optimizations:
- Adaptive frame skipping when processing falls behind
- Thread pooling for parallel processing
- Stateless detection logic for improved thread safety

## Notes

- Currently, the major bottleneck in system speed is YOLO detection latency, especially on high-resolution frames. So, to significantly reduce detection time, you can convert the YOLOv12 model to TensorRT format. TensorRT optimizes inference on NVIDIA GPUs and is ideal for deployment. But ensure that your GPU supports TensorRT because unfortunately mine doesn't.
- Using more diverse and annotated training data will significantly improve the detection model’s accuracy and robustness in real-world scenarios.
