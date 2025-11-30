import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
import re
import time

st.set_page_config(page_title="AI Fitness Assistant", layout="centered")

# ----------------- Dark Mode Colors -----------------
bg_color = "#0E1117"
bg_secondary = "#1E1E2E"
accent_primary = "#007AFF"
accent_secondary = "#5E5CE6"
success_color = "#34C759"
warning_color = "#FF9500"
card_bg = "#1C1C1E"
text_primary = "#FFFFFF"
text_secondary = "#A0A0A0"

# ----------------- Custom CSS -----------------
st.markdown(
    f"""
    <style>
        /* Global Styles */
        body {{
            background-color: {bg_color};
            color: {text_primary};
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }}
        
        /* Header Styling */
        h1 {{
            background: linear-gradient(135deg, {accent_primary} 0%, {accent_secondary} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3rem !important;
            margin-bottom: 1.5rem !important;
        }}
        
        h2, h3 {{
            color: {text_primary};
            font-weight: 600;
        }}

        /* Card Styling */
        .stForm, .stExpander {{
            background: {card_bg};
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}

        /* Chat wrapper */
        .chat-wrapper {{
            border-radius: 20px;
            padding: 24px;
            max-width: 900px;
            margin: 20px auto;
            background: {card_bg};
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
        }}

        /* Scrollable area */
        .chat-scroll {{
            max-height: 60vh;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
            padding: 10px;
            margin-bottom: 15px;
        }}
        
        /* Custom scrollbar */
        .chat-scroll::-webkit-scrollbar {{
            width: 8px;
        }}
        .chat-scroll::-webkit-scrollbar-track {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
        }}
        .chat-scroll::-webkit-scrollbar-thumb {{
            background: linear-gradient(180deg, {accent_primary}, {accent_secondary});
            border-radius: 10px;
        }}
        .chat-scroll::-webkit-scrollbar-thumb:hover {{
            background: {accent_primary};
        }}

        /* Chat row containers */
        .chat-row {{
            display: flex;
            width: 100%;
        }}
        .chat-row.user {{
            justify-content: flex-end;
        }}
        .chat-row.bot {{
            justify-content: flex-start;
        }}

        /* Chat bubbles */
        .chat-bubble {{
            padding: 14px 18px;
            border-radius: 18px;
            display: inline-block;
            max-width: 75%;
            word-wrap: break-word;
            font-size: 15px;
            line-height: 1.5;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .chat-bubble:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        }}
        .chat-bubble.user {{
            background: linear-gradient(135deg, {accent_primary} 0%, {accent_secondary} 100%);
            color: {text_primary};
            border-bottom-right-radius: 6px;
        }}
        .chat-bubble.bot {{
            background: {card_bg};
            color: {text_primary};
            border-bottom-left-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        /* Typing animation */
        .typing-bubble {{
            align-self: flex-start;
            display: flex;
            align-items: center;
            background: {card_bg};
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 18px;
            padding: 10px 14px;
            height: 35px;
            width: 65px;
            justify-content: space-around;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}
        .dot {{
            width: 8px;
            height: 8px;
            background: #8E8E93;
            border-radius: 50%;
            animation: blink 1.4s infinite both;
        }}
        .dot:nth-child(2) {{
            animation-delay: 0.2s;
        }}
        .dot:nth-child(3) {{
            animation-delay: 0.4s;
        }}
        @keyframes blink {{
            0%, 80%, 100% {{ opacity: 0.2; }}
            40% {{ opacity: 1; }}
        }}

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 12px;
            background-color: transparent;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: {card_bg};
            color: {text_secondary};
            border-radius: 12px;
            padding: 12px 24px;
            font-weight: 600;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            background: rgba(0, 122, 255, 0.1);
            border-color: {accent_primary};
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, {accent_primary} 0%, {accent_secondary} 100%);
            color: {text_primary};
            border-color: transparent;
        }}
        
        /* Button styling */
        .stButton > button {{
            background: linear-gradient(135deg, {accent_primary} 0%, {accent_secondary} 100%);
            color: {text_primary};
            border: none;
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 122, 255, 0.3);
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 122, 255, 0.4);
        }}
        
        /* Input fields */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div,
        .stSelectbox > div > div > div,
        .stSelectbox [data-baseweb="select"] {{
            background-color: {card_bg};
            color: {text_primary};
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 0.75rem;
        }}
        
        /* Radio buttons styling */
        .stRadio > div {{
            background-color: {card_bg};
            padding: 0.5rem;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .stRadio label {{
            color: {text_primary} !important;
        }}
        
        /* Remove problematic dropdown styling */
        .stSelectbox [role="listbox"],
        .stSelectbox [role="option"] {{
            color: inherit !important;
        }}
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stNumberInput > div > div > input:focus {{
            border-color: {accent_primary};
            box-shadow: 0 0 0 2px rgba(0, 122, 255, 0.2);
        }}
        
        /* Success/Info boxes */
        .stSuccess {{
            background-color: rgba(52, 199, 89, 0.1);
            border-left: 4px solid {success_color};
            border-radius: 8px;
        }}
        
        .stInfo {{
            background-color: rgba(0, 122, 255, 0.1);
            border-left: 4px solid {accent_primary};
            border-radius: 8px;
        }}
        
        /* Expander styling */
        .streamlit-expanderHeader {{
            background: {card_bg};
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            font-weight: 600;
        }}
        
        /* Chat scroll button */
        .scroll-to-bottom {{
            position: fixed;
            bottom: 120px;
            right: 40px;
            background: linear-gradient(135deg, {accent_primary} 0%, {accent_secondary} 100%);
            color: white;
            border: none;
            border-radius: 50%;
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0, 122, 255, 0.4);
            transition: transform 0.2s ease;
            z-index: 1000;
        }}
        .scroll-to-bottom:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0, 122, 255, 0.5);
        }}
        
        /* Avatar styles */
        .avatar {{
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 14px;
            flex-shrink: 0;
        }}
        .avatar-user {{
            background: linear-gradient(135deg, {accent_primary} 0%, {accent_secondary} 100%);
            color: white;
        }}
        .avatar-bot {{
            background: {card_bg};
            border: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 20px;
        }}
        input:focus, textarea:focus, select:focus, button:focus {{
            outline: 2px solid {accent_primary} !important;
            outline-offset: 2px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- Load Model -----------------
@st.cache_resource
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=250,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    return pipe

pipe = load_model()

# ----------------- Initialize Session State -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "profile" not in st.session_state:
    st.session_state.profile = {
        "name": "",
        "age": "",
        "weight": "",
        "height": "",
        "fitness_goal": "",
        "experience_level": "",
        "injuries": "",
        "preferences": ""
    }

if "workouts" not in st.session_state:
    st.session_state.workouts = []

if "generated_plan" not in st.session_state:
    st.session_state.generated_plan = None

# ----------------- Title & Tabs -----------------
st.title("💪 AI Fitness Assistant")
tab1, tab2, tab3, tab4 = st.tabs(["Profile", "Workout Log", "Achievements", "Chat"])

# ----------------- PROFILE TAB -----------------
with tab1:
    col_header, col_clear = st.columns([4, 1])
    with col_header:
        st.header("👤 Your Profile")
    with col_clear:
        if st.button("🗑️ Clear", key="clear_profile", help="Clear all profile data"):
            st.session_state.profile = {
                "name": "",
                "age": "",
                "weight": "",
                "height": "",
                "fitness_goal": "",
                "experience_level": "",
                "injuries": "",
                "preferences": ""
            }
            st.rerun()
    
    # Profile completion indicator
    profile = st.session_state.profile
    filled_fields = sum(1 for v in profile.values() if v)
    total_fields = len(profile)
    completion = int((filled_fields / total_fields) * 100)
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #007AFF 0%, #007AFF {completion}%, rgba(255,255,255,0.1) {completion}%, rgba(255,255,255,0.1) 100%); 
                height: 8px; border-radius: 4px; margin-bottom: 1rem;"></div>
    <p style="color: #A0A0A0; font-size: 0.9rem; margin-top: 0.5rem;">Profile {completion}% complete</p>
    """, unsafe_allow_html=True)
    
    st.write("This information helps personalize your fitness recommendations.")
    
    with st.form("profile_form"):
        name = st.text_input("Name (optional)", value=st.session_state.profile["name"])
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.text_input("Age", value=st.session_state.profile["age"])
            weight = st.text_input("Weight (lbs or kg)", value=st.session_state.profile["weight"])
        with col2:
            height = st.text_input("Height", value=st.session_state.profile["height"])
            
            # Experience level with radio buttons instead of dropdown
            current_exp = st.session_state.profile["experience_level"]
            experience_level = st.radio(
                "Experience Level",
                ["Beginner", "Intermediate", "Advanced"],
                index=["Beginner", "Intermediate", "Advanced"].index(current_exp) if current_exp in ["Beginner", "Intermediate", "Advanced"] else 0,
                horizontal=True
            )
        
        fitness_goal = st.text_area(
            "Fitness Goals (e.g., lose weight, build muscle, improve endurance)",
            value=st.session_state.profile["fitness_goal"],
            height=80
        )
        
        injuries = st.text_area(
            "Injuries or Limitations (optional)",
            value=st.session_state.profile["injuries"],
            height=60
        )
        
        preferences = st.text_area(
            "Preferences (e.g., gym vs home, equipment available, time constraints)",
            value=st.session_state.profile["preferences"],
            height=80
        )
        
        submitted = st.form_submit_button("Save Profile")
        
        if submitted:
            st.session_state.profile = {
                "name": name,
                "age": age,
                "weight": weight,
                "height": height,
                "fitness_goal": fitness_goal,
                "experience_level": experience_level,
                "injuries": injuries,
                "preferences": preferences
            }
            st.success("✅ Profile saved!")

# ----------------- CHAT TAB -----------------
with tab4:
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    
    # Chat container with messages
    if st.session_state.messages:
        # Header with clear button
        col1, col2 = st.columns([5, 1])
        with col2:
            if st.button("🗑️ Clear", key="clear_chat", help="Clear conversation"):
                st.session_state.messages = []
                st.rerun()
        
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        
        # Get user initials for avatar
        user_name = st.session_state.profile.get("name", "")
        if user_name:
            initials = "".join([n[0].upper() for n in user_name.split()[:2]])
        else:
            initials = ""
        
        # Messages display with scroll container and auto-scroll
        st.markdown('''
        <div class="chat-container" id="chat-container" style="max-height: 60vh; overflow-y: auto; padding-right: 10px;">
        <script>
            // Auto-scroll to bottom when new messages arrive
            setTimeout(function() {
                var container = document.getElementById('chat-container');
                if (container) {
                    container.scrollTop = container.scrollHeight;
                }
            }, 100);
        </script>
        ''', unsafe_allow_html=True)
        
        for role, message in st.session_state.messages:
            safe_message = (
                message.replace("&", "&amp;")
                       .replace("<", "&lt;")
                       .replace(">", "&gt;")
                       .replace("\n", "<br>")
            )
            
            if role == "user":
                avatar_html = f'<div class="avatar avatar-user">{initials if initials else ""}</div>' if initials else '<div class="avatar avatar-user" style="background: #000000;"></div>'
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-end; align-items: flex-end; gap: 10px; margin-bottom: 16px;">
                    <div style="background: linear-gradient(135deg, #007AFF 0%, #5E5CE6 100%);
                                color: white;
                                padding: 12px 16px;
                                border-radius: 18px;
                                border-bottom-right-radius: 4px;
                                max-width: 70%;
                                box-shadow: 0 2px 8px rgba(0, 122, 255, 0.3);
                                word-wrap: break-word;">
                        {safe_message}
                    </div>
                    {avatar_html}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-start; align-items: flex-end; gap: 10px; margin-bottom: 16px;">
                    <div class="avatar avatar-bot">🏋️</div>
                    <div style="background: #2C2C2E;
                                color: #FFFFFF;
                                padding: 12px 16px;
                                border-radius: 18px;
                                border-bottom-left-radius: 4px;
                                max-width: 70%;
                                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
                                border: 1px solid rgba(255, 255, 255, 0.1);
                                word-wrap: break-word;">
                        {safe_message}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Check if we need to show typing indicator
        needs_response = st.session_state.messages and st.session_state.messages[-1][0] == "user"
        
        if needs_response:
            st.markdown("""
            <div style="display: flex; justify-content: flex-start; align-items: flex-end; gap: 10px; margin-bottom: 16px;">
                <div class="avatar avatar-bot">🏋️</div>
                <div style="background: #2C2C2E;
                            border: 1px solid rgba(255, 255, 255, 0.1);
                            border-radius: 18px;
                            padding: 12px 16px;
                            display: flex;
                            gap: 6px;
                            align-items: center;
                            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Empty state
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; color: #A0A0A0;">
            <h3 style="color: #FFFFFF; margin-bottom: 10px;">👋 Start a conversation</h3>
            <p>Ask me anything about fitness, workouts, or nutrition!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Input at the bottom
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    prompt = st.chat_input("Ask me anything about fitness...")

    if prompt:
        st.session_state.messages.append(("user", prompt))
        st.rerun()

    # Generate Response
    needs_response = st.session_state.messages and st.session_state.messages[-1][0] == "user"
    if needs_response:
        time.sleep(1.2)
        
        last_prompt = st.session_state.messages[-1][1]

        # Build conversation history for context
        conversation_history = []
        # Include last 3 exchanges for context (6 messages total)
        start_index = max(0, len(st.session_state.messages) - 7)
        for role, msg in st.session_state.messages[start_index:-1]:
            conversation_history.append(f"{'User' if role == 'user' else 'Assistant'}: {msg}")
        
        context_str = "\n".join(conversation_history) if conversation_history else ""
        
        # Build profile context
        profile = st.session_state.profile
        profile_info = []
        
        if profile["name"]:
            profile_info.append(f"The user's name is {profile['name']}")
        if profile["age"]:
            profile_info.append(f"they are {profile['age']} years old")
        if profile["experience_level"]:
            profile_info.append(f"experience level is {profile['experience_level'].lower()}")
        if profile["fitness_goal"]:
            profile_info.append(f"their goal: {profile['fitness_goal']}")
        if profile["injuries"]:
            profile_info.append(f"injury considerations: {profile['injuries']}")
        if profile["preferences"]:
            profile_info.append(f"preferences: {profile['preferences']}")
        
        profile_context = "; ".join(profile_info) + "." if profile_info else ""

        # Add workout history context
        recent_workouts = st.session_state.workouts[-5:] if st.session_state.workouts else []
        workout_context = ""
        if recent_workouts:
            workout_summary = [
                f"{w['exercise']} {w['sets']}x{w['reps']} @ {w['weight']}lbs on {w['date']}"
                for w in recent_workouts
            ]
            workout_context = " Recent workouts: " + "; ".join(workout_summary) + "."

        # Define system prompt for Mistral
        system_prompt = (
            f"You are a helpful fitness coach. {profile_context}{workout_context} "
            "Answer briefly and stop when done. For simple questions, give 1-2 sentences. "
            "For workout advice, suggest 2-3 exercises with one sentence each explaining how to do it."
        )
        
        # Build full prompt with conversation history
        if context_str:
            prompt_wrapped = f"<s>[INST] {system_prompt}\n\nPrevious conversation:\n{context_str}\n\nCurrent question: {last_prompt.strip()} [/INST]"
        else:
            prompt_wrapped = f"<s>[INST] {system_prompt}\n\n{last_prompt.strip()} [/INST]"

        # Generate model response - Focus on natural stopping
        raw_output = pipe(
            prompt_wrapped,
            max_new_tokens=180,
            temperature=0.7,
            top_p=0.85,
            top_k=40,
            do_sample=True,
            repetition_penalty=1.3,
            pad_token_id=pipe.tokenizer.eos_token_id,
            eos_token_id=pipe.tokenizer.eos_token_id,
        )[0]["generated_text"]

        # Clean the output for Mistral
        response = raw_output
        
        # Remove Mistral special tokens
        response = re.sub(r"<s>|</s>|\[INST\]|\[/INST\]", "", response)
        
        # Remove the ENTIRE input (system + user question) from the start
        # Find where the actual response begins after [/INST]
        inst_end = raw_output.find("[/INST]")
        if inst_end != -1:
            response = raw_output[inst_end + 7:].strip()
        
        # Aggressively remove any echoed question at the start
        # This handles cases where the question appears with or without punctuation
        question_clean = last_prompt.strip().rstrip('?!.')
        response = re.sub(
            f"^{re.escape(question_clean)}[?!.]*\\s*",
            "",
            response,
            flags=re.IGNORECASE
        )
        
        # Remove system prompt remnants
        response = re.sub(
            r"You are a helpful fitness coach\..*?unless asked for detailed plans\.",
            "",
            response,
            flags=re.DOTALL | re.IGNORECASE
        )
        
        # Stop at any meta-text like "Question:", "Answer:", or profile repetition
        stop_patterns = [
            "Question:", "Answer:", "\nUser:", "\nQuestion", "Q:", "\nA:", 
            "User:", "Assistant:", "What is your", "What are your",
            "Your experience level", "Your goal is", "Given your"
        ]
        for stop_word in stop_patterns:
            if stop_word in response:
                response = response.split(stop_word)[0]
        
        # Check if it's a simple greeting/short question - enforce brevity
        simple_patterns = ['how are you', 'hello', 'hi', 'hey', 'thanks', 'thank you', 'yes', 'no', 'okay', 'ok', 'doing', 'how old', 'what is my', 'who am i']
        is_simple = any(pattern in last_prompt.lower() for pattern in simple_patterns)
        
        # Always limit to prevent runaway responses
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', response.strip())
        
        if is_simple and len(sentences) > 2:
            response = ' '.join(sentences[:2])
        elif len(sentences) > 5:
            # Cap at 5 sentences max for any response
            response = ' '.join(sentences[:5])
        
        # Clean up whitespace
        response = response.strip()
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = re.sub(r' {2,}', ' ', response)

        if not response:
            response = "Let's try that again — could you rephrase your question?"

        st.session_state.messages.append(("bot", response))
        st.rerun()

# ----------------- WORKOUT LOG TAB -----------------
with tab2:
    col_header, col_clear = st.columns([4, 1])
    with col_header:
        st.header("📋 Workout Log")
    with col_clear:
        if st.session_state.workouts:
            if st.button("🗑️ Clear Log", key="clear_workouts"):
                st.session_state.workouts = []
                st.session_state.generated_plan = None
                st.rerun()
    
    st.write("Track your workouts and let the AI adjust recommendations based on your progress.")
    
    # Generate workout plan section
    st.subheader("🤖 AI Generated Workout")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        plan_prompt = st.text_input(
            "Ask AI to create a workout plan",
            placeholder="e.g., 'Create a 3-day strength training plan' or 'Give me a leg day workout'"
        )
    with col2:
        st.write("")
        st.write("")
        generate_btn = st.button("Generate Plan", type="primary")
    
    if generate_btn and plan_prompt:
        with st.spinner("🏋️ Creating your workout plan..."):
            # Build context for workout generation
            profile = st.session_state.profile
            profile_info = []
            
            if profile["name"]:
                profile_info.append(f"Name: {profile['name']}")
            if profile["age"]:
                profile_info.append(f"Age: {profile['age']}")
            if profile["experience_level"]:
                profile_info.append(f"Experience: {profile['experience_level']}")
            if profile["fitness_goal"]:
                profile_info.append(f"Goal: {profile['fitness_goal']}")
            if profile["injuries"]:
                profile_info.append(f"Injuries: {profile['injuries']}")
            
            profile_context = "; ".join(profile_info) if profile_info else ""
            
            # Get recent workout history
            recent_workouts = st.session_state.workouts[-5:] if st.session_state.workouts else []
            workout_context = ""
            if recent_workouts:
                workout_context = "Recent workouts: " + "; ".join([
                    f"{w['exercise']} {w['weight']}lbs {w['sets']}x{w['reps']}"
                    for w in recent_workouts
                ])
            
            system_prompt = (
                f"You are a fitness coach creating a workout plan. {profile_context}. {workout_context}. "
                "Create 3-4 exercises. Format each as: Exercise Name | X sets | Y reps | Z lbs | Brief tip\n"
                "Be specific with weights. Keep tips short (5-8 words). Complete all exercises."
            )
            
            prompt_wrapped = f"<s>[INST] {system_prompt}\n\n{plan_prompt.strip()} [/INST]"
            
            # Workout plan generation - Balanced token limit
            raw_output = pipe(
                prompt_wrapped,
                max_new_tokens=280,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                eos_token_id=pipe.tokenizer.eos_token_id
            )[0]["generated_text"]
            
            # Clean output
            response = raw_output
            inst_end = raw_output.find("[/INST]")
            if inst_end != -1:
                response = raw_output[inst_end + 7:].strip()
            
            response = re.sub(r"<s>|</s>|\[INST\]|\[/INST\]", "", response)
            
            # Remove any system prompt echoes
            response = re.sub(
                r"You are a fitness coach.*?Be specific with weights\.",
                "",
                response,
                flags=re.DOTALL | re.IGNORECASE
            )
            
            response = response.strip()
            
            st.session_state.generated_plan = response
    
    # Display generated plan
    if st.session_state.generated_plan:
        st.success("✅ Workout Plan Generated!")
        st.markdown("---")
        st.markdown(st.session_state.generated_plan)
        st.markdown("---")
        
        # Parse and add to log button
        if st.button("➕ Add This Plan to Workout Log", key="add_plan"):
            # Parse the generated plan
            lines = st.session_state.generated_plan.strip().split('\n')
            added_count = 0
            
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Look for lines with the pipe separator
                if '|' in line:
                    # Remove number prefix like "1. " or "1)" or just "1"
                    line = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
                    
                    # Split by pipe
                    parts = [p.strip() for p in line.split('|')]
                    
                    if len(parts) >= 4:
                        exercise_name = parts[0].strip()
                        sets_str = parts[1].lower().replace('sets', '').replace('set', '').strip()
                        reps_str = parts[2].lower().replace('reps', '').replace('rep', '').strip()
                        weight_str = parts[3].lower().replace('lbs', '').replace('lb', '').strip()
                        notes = parts[4].strip() if len(parts) > 4 else ""
                        
                        try:
                            # Parse sets - handle "3" or "3 sets" or "3-4"
                            sets_match = re.search(r'(\d+)', sets_str)
                            sets = int(sets_match.group(1)) if sets_match else 3
                            
                            # Parse reps - handle "8-10" or "10" or "8-10 reps"
                            reps_match = re.search(r'(\d+)', reps_str)
                            reps = int(reps_match.group(1)) if reps_match else 10
                            
                            # Parse weight - handle "135" or "135 lbs" or "135-145"
                            weight_match = re.search(r'(\d+)', weight_str)
                            weight = int(weight_match.group(1)) if weight_match else 0
                            
                            # Only add if we have a valid exercise name
                            if exercise_name:
                                st.session_state.workouts.append({
                                    "date": time.strftime("%Y-%m-%d"),
                                    "exercise": exercise_name,
                                    "sets": sets,
                                    "reps": reps,
                                    "weight": weight,
                                    "notes": notes,
                                    "completed": False
                                })
                                added_count += 1
                        except Exception as e:
                            # Skip exercises that fail to parse
                            st.warning(f"Couldn't parse: {line}")
                            continue
            
            if added_count > 0:
                st.success(f"✅ Added {added_count} exercises to your workout log!")
                st.session_state.generated_plan = None
                st.rerun()
            else:
                st.error("❌ Couldn't parse any exercises. Try regenerating the plan.")
    
    st.markdown("---")
    
    # Manual workout entry
    st.subheader("📝 Log a Workout Manually")
    
    with st.form("workout_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            exercise = st.text_input("Exercise Name", placeholder="e.g., Barbell Squat")
            sets = st.number_input("Sets", min_value=1, max_value=10, value=3)
            reps = st.number_input("Reps", min_value=1, max_value=50, value=10)
        
        with col2:
            weight = st.number_input("Weight (lbs)", min_value=0, max_value=1000, value=135)
            workout_date = st.date_input("Date", value=time.strftime("%Y-%m-%d"))
        
        notes = st.text_area("Notes (optional)", placeholder="How did it feel? Any modifications?")
        
        submitted = st.form_submit_button("Log Workout")
        
        if submitted and exercise:
            st.session_state.workouts.append({
                "date": str(workout_date),
                "exercise": exercise,
                "sets": sets,
                "reps": reps,
                "weight": weight,
                "notes": notes,
                "completed": True
            })
            st.success(f"✅ Logged: {exercise} - {sets}x{reps} @ {weight}lbs")
            st.rerun()
    
    st.markdown("---")
    
    # Display workout history
    st.subheader("📊 Workout History")
    
    if st.session_state.workouts:
        # Show some stats
        total_workouts = len(st.session_state.workouts)
        completed_workouts = sum(1 for w in st.session_state.workouts if w.get("completed", False))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Exercises", total_workouts)
        with col2:
            st.metric("Completed", completed_workouts)
        with col3:
            completion_rate = int((completed_workouts / total_workouts) * 100) if total_workouts > 0 else 0
            st.metric("Completion Rate", f"{completion_rate}%")
        
        st.markdown("---")
        
        # Group by date
        from collections import defaultdict
        workouts_by_date = defaultdict(list)
        for workout in st.session_state.workouts:
            workouts_by_date[workout["date"]].append(workout)
        
        # Display in reverse chronological order
        for date_idx, date in enumerate(sorted(workouts_by_date.keys(), reverse=True)):
            completed_count = sum(1 for w in workouts_by_date[date] if w.get("completed", False))
            total_count = len(workouts_by_date[date])
            
            # Keep today's workout expanded by default
            is_today = date == time.strftime("%Y-%m-%d")
            
            with st.expander(f"📅 {date} • {completed_count}/{total_count} completed", expanded=is_today):
                for i, workout in enumerate(workouts_by_date[date]):
                    status = "✅" if workout.get("completed", False) else "⏳"
                    
                    # Exercise card
                    st.markdown(f"""
                    <div style="background: rgba(0, 122, 255, 0.05); padding: 1rem; border-radius: 12px; margin-bottom: 1rem; border-left: 3px solid {'#34C759' if workout.get('completed') else '#FF9500'};">
                        <h4 style="margin: 0; color: #FFFFFF;">{status} {workout['exercise']}</h4>
                        <p style="color: #A0A0A0; margin: 0.5rem 0;">
                            <strong>{workout['sets']}</strong> sets × <strong>{workout['reps']}</strong> reps @ <strong>{workout['weight']}</strong> lbs
                        </p>
                        {f"<p style='color: #A0A0A0; font-style: italic; margin: 0;'>{workout['notes']}</p>" if workout['notes'] else ""}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Action buttons
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        if not workout.get("completed", False):
                            if st.button(f"✓ Mark Complete", key=f"complete_{date}_{i}"):
                                workout["completed"] = True
                                # Don't rerun - just update the state
                    with col2:
                        new_weight = st.number_input(
                            f"Update weight for {workout['exercise']}",
                            value=workout["weight"],
                            key=f"weight_{date}_{i}",
                            label_visibility="collapsed"
                        )
                        if new_weight != workout["weight"]:
                            if st.button("💾 Save", key=f"update_{date}_{i}"):
                                workout["weight"] = new_weight
                    with col3:
                        if st.button("🗑️", key=f"delete_{date}_{i}", help="Delete exercise"):
                            st.session_state.workouts.remove(workout)
                            st.rerun()
    else:
        st.info("💡 No workouts logged yet. Generate a plan or log a workout manually to get started!")

# ----------------- ACHIEVEMENTS TAB -----------------
with tab3:
    st.header("🏆 Achievements")
    
    if st.session_state.workouts:
        # Calculate stats
        from collections import defaultdict
        from datetime import datetime
        
        # Get personal records (max weight per exercise)
        pr_dict = defaultdict(lambda: {"weight": 0, "date": "", "sets": 0, "reps": 0})
        
        for workout in st.session_state.workouts:
            exercise = workout["exercise"]
            weight = workout["weight"]
            if weight > pr_dict[exercise]["weight"]:
                pr_dict[exercise] = {
                    "weight": weight,
                    "date": workout["date"],
                    "sets": workout["sets"],
                    "reps": workout["reps"]
                }
        
        # Calculate other stats
        total_workouts = len(st.session_state.workouts)
        completed_workouts = sum(1 for w in st.session_state.workouts if w.get("completed", False))
        unique_exercises = len(set(w["exercise"] for w in st.session_state.workouts))
        
        # Get unique workout dates
        workout_dates = set(w["date"] for w in st.session_state.workouts)
        days_worked_out = len(workout_dates)
        
        # Calculate total volume (sets × reps × weight)
        total_volume = sum(w["sets"] * w["reps"] * w["weight"] for w in st.session_state.workouts)
        
        # Top stats cards
        st.subheader("📊 Your Stats")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #007AFF 0%, #5E5CE6 100%); 
                        padding: 1.5rem; border-radius: 16px; text-align: center;">
                <h2 style="margin: 0; color: white; font-size: 2.5rem;">{days_worked_out}</h2>
                <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9);">Days Trained</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #34C759 0%, #30D158 100%); 
                        padding: 1.5rem; border-radius: 16px; text-align: center;">
                <h2 style="margin: 0; color: white; font-size: 2.5rem;">{unique_exercises}</h2>
                <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9);">Unique Exercises</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #FF9500 0%, #FF9F0A 100%); 
                        padding: 1.5rem; border-radius: 16px; text-align: center;">
                <h2 style="margin: 0; color: white; font-size: 2.5rem;">{completed_workouts}</h2>
                <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9);">Completed</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #FF375F 0%, #FF453A 100%); 
                        padding: 1.5rem; border-radius: 16px; text-align: center;">
                <h2 style="margin: 0; color: white; font-size: 2.5rem;">{total_volume:,}</h2>
                <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9);">Total Volume</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Personal Records section
        st.subheader("🏋️ Personal Records")
        
        # Search bar
        search_query = st.text_input("🔍 Search exercises", placeholder="e.g., Squat, Bench Press, Deadlift...")
        
        # Filter PRs based on search
        filtered_prs = {
            exercise: data 
            for exercise, data in pr_dict.items() 
            if search_query.lower() in exercise.lower()
        } if search_query else pr_dict
        
        if filtered_prs:
            # Sort by weight descending
            sorted_prs = sorted(filtered_prs.items(), key=lambda x: x[1]["weight"], reverse=True)
            
            # Display as cards
            for i in range(0, len(sorted_prs), 2):
                col1, col2 = st.columns(2)
                
                # First card
                with col1:
                    exercise, data = sorted_prs[i]
                    st.markdown(f"""
                    <div style="background: {card_bg}; 
                                padding: 1.5rem; 
                                border-radius: 16px; 
                                border: 1px solid rgba(255, 255, 255, 0.1);
                                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <h3 style="margin: 0; color: #FFFFFF;">🏆 {exercise}</h3>
                        </div>
                        <div style="background: linear-gradient(135deg, #007AFF 0%, #5E5CE6 100%);
                                    padding: 1rem;
                                    border-radius: 12px;
                                    text-align: center;
                                    margin-bottom: 1rem;">
                            <h2 style="margin: 0; color: white; font-size: 2.5rem;">{data['weight']} lbs</h2>
                        </div>
                        <p style="color: #A0A0A0; margin: 0.25rem 0;">
                            <strong>{data['sets']}</strong> sets × <strong>{data['reps']}</strong> reps
                        </p>
                        <p style="color: #A0A0A0; margin: 0.25rem 0; font-size: 0.9rem;">
                            📅 {data['date']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Second card (if exists)
                if i + 1 < len(sorted_prs):
                    with col2:
                        exercise, data = sorted_prs[i + 1]
                        st.markdown(f"""
                        <div style="background: {card_bg}; 
                                    padding: 1.5rem; 
                                    border-radius: 16px; 
                                    border: 1px solid rgba(255, 255, 255, 0.1);
                                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                <h3 style="margin: 0; color: #FFFFFF;">🏆 {exercise}</h3>
                            </div>
                            <div style="background: linear-gradient(135deg, #007AFF 0%, #5E5CE6 100%);
                                        padding: 1rem;
                                        border-radius: 12px;
                                        text-align: center;
                                        margin-bottom: 1rem;">
                                <h2 style="margin: 0; color: white; font-size: 2.5rem;">{data['weight']} lbs</h2>
                            </div>
                            <p style="color: #A0A0A0; margin: 0.25rem 0;">
                                <strong>{data['sets']}</strong> sets × <strong>{data['reps']}</strong> reps
                            </p>
                            <p style="color: #A0A0A0; margin: 0.25rem 0; font-size: 0.9rem;">
                                📅 {data['date']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Show search results count
            if search_query:
                st.info(f"Found {len(filtered_prs)} exercise(s) matching '{search_query}'")
        else:
            if search_query:
                st.warning(f"No exercises found matching '{search_query}'")
            else:
                st.info("No personal records yet. Start logging workouts to track your progress!")
        
        st.markdown("---")
        
        # Recent milestones
        st.subheader("🎯 Recent Activity")
        recent_workouts = sorted(st.session_state.workouts, key=lambda x: x["date"], reverse=True)[:5]
        
        for workout in recent_workouts:
            status = "✅" if workout.get("completed", False) else "⏳"
            st.markdown(f"""
            <div style="background: rgba(0, 122, 255, 0.05); 
                        padding: 1rem; 
                        border-radius: 12px; 
                        margin-bottom: 0.5rem;
                        border-left: 3px solid {'#34C759' if workout.get('completed') else '#FF9500'};">
                <strong>{status} {workout['exercise']}</strong> - {workout['weight']} lbs × {workout['sets']}x{workout['reps']}
                <span style="color: #A0A0A0; float: right;">📅 {workout['date']}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("💡 No achievements yet! Start logging workouts to track your progress and personal records.")