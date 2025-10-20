#!/bin/bash
# Mirror of Erised Startup Script for Raspberry Pi 5
# Optimized for 10-inch LCD (1024x600)

echo "🔮 Starting Mirror of Erised on Raspberry Pi 5..."

# Set display environment variables for better performance
export DISPLAY=:0
export SDL_VIDEO_WINDOW_POS=0,0
export SDL_VIDEO_CENTERED=0

# Optional: Disable screen blanking for continuous display
# xset s off
# xset -dpms
# xset s noblank

# Run the Mirror of Erised with Pi 5 optimizations
# For development (windowed mode):
python3 main.py \
    --display-width 1024 \
    --display-height 600 \
    --images-dir emotion_images \
    --whisper-model tiny \
    --piper-voice en_GB-alan-medium.onnx

# For production (fullscreen mode), uncomment the line below and comment the above:
# python3 main.py --fullscreen --display-width 1024 --display-height 600 --images-dir emotion_images --whisper-model tiny --piper-voice en_GB-alan-medium.onnx

echo "🔮 Mirror of Erised stopped."

