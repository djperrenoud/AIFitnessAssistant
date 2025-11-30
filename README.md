# AI Fitness Assistant

This project is a Streamlit-based AI Fitness Assistant powered by a locally loaded, quantized Mistral-7B model.  
The application includes a user profile tab, an AI chat interface, workout logging, and a custom dark UI.

---

## Features

- User Profile input (age, weight, height, goals, experience level, injuries, etc.)
- AI Chat that uses your profile and workout history as context
- Workout Log (exercise name, sets, reps, weight, date)
- Achievements section
- Local model loading using 4-bit quantization
- No API keys required

---

### Installation & Run

```bash
# Clone the repository
git clone https://github.com/djperrenoud/AIFitnessAssistant.git
cd AIFitnessAssistant

# Install dependencies
pip install -r requirements.txt

# Start the app
streamlit run app.py
