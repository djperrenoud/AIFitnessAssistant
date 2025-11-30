# AI Fitness Assistant

This project is a Streamlit-based AI Fitness Assistant powered by a locally loaded, quantized Mistral-7B model.  
The application includes a user profile tab, an AI chat interface, workout logging, and a clean custom dark UI.

---

## Features

- User Profile input (age, weight, height, goals, experience level, injuries, etc.)
- AI Chat that uses your profile and workout history as context
- Workout Log (exercise name, sets, reps, weight, date)
- Achievements section (placeholder for future updates)
- Local model loading using 4-bit quantization
- No API keys required

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

pip install -r requirements.txt

streamlit run app.py

http://localhost:8501
