import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="centered"
)

# Medical knowledge database
medical_data = [
    {
        "question": "What is diabetes?",
        "keywords": "diabetes blood sugar insulin glucose",
        "answer": """
        Diabetes is a chronic health condition that affects
        how the body regulates blood glucose.

        Main types:
        1. Type 1 Diabetes
        2. Type 2 Diabetes
        3. Gestational Diabetes

        Symptoms may include increased thirst, frequent
        urination and tiredness.

        Consult a healthcare professional for diagnosis
        and treatment.
        """
    },
    {
        "question": "What is hypertension?",
        "keywords": "hypertension high blood pressure heart",
        "answer": """
        Hypertension is a condition in which blood pressure
        remains higher than the recommended level.

        Risk factors include:
        1. High salt intake
        2. Lack of physical activity
        3. Family history
        4. Certain health conditions

        Regular blood pressure monitoring is important.
        """
    },
    {
        "question": "What is anemia?",
        "keywords": "anemia haemoglobin iron red blood cells",
        "answer": """
        Anemia is a condition in which the blood has a
        lower-than-normal amount of healthy red blood cells
        or haemoglobin.

        Common symptoms include tiredness and weakness.

        Iron deficiency is one possible cause.
        Medical evaluation is recommended.
        """
    },
    {
        "question": "What is malaria?",
        "keywords": "malaria mosquito parasite plasmodium fever",
        "answer": """
        Malaria is an infectious disease caused by
        Plasmodium parasites.

        It spreads through bites from infected female
        Anopheles mosquitoes.

        Common symptoms include fever, chills and headache.

        Early medical diagnosis and treatment are important.
        """
    },
    {
        "question": "What is tuberculosis?",
        "keywords": "tuberculosis TB lungs bacteria infection cough",
        "answer": """
        Tuberculosis (TB) is an infectious disease caused
        by bacteria called Mycobacterium tuberculosis.

        It commonly affects the lungs.

        Symptoms may include prolonged cough, fever,
        night sweats and weight loss.

        Consult a doctor if TB is suspected.
        """
    },
    {
        "question": "How can we prevent infections?",
        "keywords": "infection prevention hygiene handwashing germs",
        "answer": """
        General infection prevention methods include:

        1. Wash hands regularly.
        2. Maintain personal hygiene.
        3. Follow recommended vaccinations.
        4. Handle food safely.
        5. Follow respiratory hygiene.
        6. Keep surroundings clean.
        """
    },
    {
        "question": "What is the importance of vitamins?",
        "keywords": "vitamins nutrition vitamin deficiency health",
        "answer": """
        Vitamins are essential nutrients required for
        normal growth and body functions.

        Examples:
        Vitamin A - Supports vision.
        Vitamin B - Helps energy metabolism.
        Vitamin C - Supports normal immune function.
        Vitamin D - Helps maintain bone health.
        Vitamin K - Helps blood clotting.
        """
    },
    {
        "question": "What is dehydration?",
        "keywords": "dehydration water fluid loss thirst",
        "answer": """
        Dehydration occurs when the body loses more fluid
        than it takes in.

        Possible signs include thirst, dry mouth,
        dizziness and reduced urination.

        Drinking appropriate fluids can help prevent
        dehydration. Severe symptoms need medical attention.
        """
    }
]

# Prepare text for NLP
documents = [
    item["question"] + " " + item["keywords"]
    for item in medical_data
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)


# Function to find the best answer
def get_medical_answer(user_question):
    query_vector = vectorizer.transform([user_question])

    similarity = cosine_similarity(
        query_vector,
        tfidf_matrix
    )

    best_match = similarity.argmax()
    confidence = similarity[0][best_match]

    if confidence < 0.15:
        return (
            "Sorry, I couldn't find relevant information "
            "in my medical knowledge database. "
            "Please consult a qualified healthcare professional."
        )

    return medical_data[best_match]["answer"]


# Application interface
st.title("🏥 AI-Powered Medical Knowledge Assistant")

st.write(
    "Welcome! Ask questions about common diseases, "
    "health conditions and medical knowledge."
)

st.info(
    "This application provides general educational "
    "information only. It cannot diagnose diseases "
    "or replace professional medical advice."
)

# Sidebar
st.sidebar.title("About the Project")

st.sidebar.write(
    "An NLP-based medical knowledge assistant "
    "that retrieves relevant information using "
    "TF-IDF and cosine similarity."
)

st.sidebar.subheader("Available Topics")

for item in medical_data:
    st.sidebar.write("• " + item["question"])

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input(
    "Ask your medical question here..."
)

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    answer = get_medical_answer(user_input)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()

# Footer
st.divider()

st.caption(
    "AI Medical Assistant | Educational Project | "
    "Not a substitute for professional healthcare."
)
