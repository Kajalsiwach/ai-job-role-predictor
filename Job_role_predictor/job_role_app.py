import streamlit as st
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# ---------------- SESSION STATE ---------------- #
if "history" not in st.session_state:
    st.session_state.history = []
if "best_role" not in st.session_state:
    st.session_state.best_role = ""
if "confidence" not in st.session_state:
    st.session_state.confidence = 0
if "skills_input" not in st.session_state:
    st.session_state.skills_input = ""
if "recommended_skills" not in st.session_state:
    st.session_state.recommended_skills = []

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AI Job Role Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ---------------- #
current_folder = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_folder, "model", "job_model.pkl")
vectorizer_path = os.path.join(current_folder, "model", "tfidf_vectorizer.pkl")
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# ---------------- THEME ---------------- #
theme = st.sidebar.radio("Choose Theme", ["Light Mode", "Dark Mode"])
if theme == "Dark Mode":
    bg = "#0f172a"
    text = "white"
    card = "#1e293b"
else:
    bg = "#f3f4f6"
    text = "black"
    card = "white"

# ---------------- CSS ---------------- #
st.markdown(f"""
<style>
body {{background:{bg}; color:{text}; font-family: 'Segoe UI', sans-serif;}}
.card {{background:{card}; padding:25px; border-radius:15px; box-shadow:0 10px 25px rgba(0,0,0,0.1); margin-bottom:25px;}}
.hero {{text-align:center; padding:60px;}}
.hero h1 {{font-size:60px; color:#6C63FF;}}
.stButton>button {{background:#6C63FF; color:white; padding:12px 25px; font-size:18px; border-radius:10px;}}
.navbar {{position:sticky; top:0; background:{card}; padding:10px; text-align:center; box-shadow:0 2px 10px rgba(0,0,0,0.1); z-index:999;}}
.navbar a {{margin:0 15px; text-decoration:none; color:#6C63FF; font-weight:bold;}}
.navbar a:hover {{color:#4B0082;}}
</style>
""", unsafe_allow_html=True)

# ---------------- NAVBAR ---------------- #
st.markdown(f"""
<div class="navbar">
<a href="#home">Home</a>
<a href="#predict">Predict</a>
<a href="#features">Features</a>
<a href="#faq">FAQ</a>
<a href="#testimonials">Testimonials</a>
<a href="#contact">Contact</a>
</div>
""", unsafe_allow_html=True)

# ---------------- HERO ---------------- #
st.markdown("""
<div class="hero" id="home">
<h1>💼 AI Job Role Predictor</h1>
<h3>Discover your perfect career path using AI</h3>
<img src="https://img.freepik.com/free-vector/job-search-concept-illustration_114360-1465.jpg" width="850">
</div>
""", unsafe_allow_html=True)

# ---------------- MAIN PREDICTION ---------------- #
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<div class="card" id="predict">', unsafe_allow_html=True)
st.subheader("Enter Your Skills")
skills_input = st.text_area("Example: Python, Machine Learning, SQL")

example_skills = [
    "python, pandas, machine-learning","html, css, react","python, django, sql",
    "aws, docker, kubernetes","sql, excel, powerbi","python, numpy, scikit-learn",
    "python, tensorflow, deep-learning","python, pytorch, ai","python, pandas, data-visualization",
    "sql, tableau, data-analysis","excel, powerbi, dashboards","java, spring, backend",
    "java, springboot, microservices","javascript, nodejs, express","javascript, react, frontend",
    "html, css, bootstrap","html, css, tailwind","react, redux, javascript",
    "angular, typescript, frontend","vue, javascript, frontend","aws, cloud, terraform",
    "aws, lambda, serverless","azure, devops, pipelines","gcp, cloud, kubernetes",
    "docker, kubernetes, devops","git, github, ci-cd","linux, bash, scripting",
    "python, flask, restapi","python, fastapi, backend","nodejs, mongodb, backend",
    "nodejs, graphql, api","mysql, sql, database","postgresql, database, sql",
    "oracle, sql, database","python, spark, big-data","scala, spark, hadoop",
    "python, airflow, etl","python, nlp, transformers","python, computer-vision, opencv",
    "c, cpp, algorithms","c++, oop, software-design","c, embedded, microcontrollers",
    "python, automation, scripting","selenium, python, testing","java, selenium, testing"
]
selected_example = st.selectbox("Or try an example:", [""] + example_skills)
if selected_example and not skills_input:
    skills_input = selected_example

c1,c2,c3 = st.columns(3)
c4,c5,c6 = st.columns(3)
c7,c8,c9 = st.columns(3)
if c1.button("Web Dev"): skills_input="html css react javascript"
if c2.button("Data Science"): skills_input="python pandas machine learning"
if c3.button("Cloud"): skills_input="aws docker kubernetes"
if c4.button("AI"): skills_input="python tensorflow deep learning"
if c5.button("Backend"): skills_input="python django sql api"
if c6.button("Frontend"): skills_input="html css javascript react"
if c7.button("DevOps"): skills_input="docker kubernetes ci cd"
if c8.button("Data Analyst"): skills_input="sql excel powerbi tableau"
if c9.button("Machine Learning"): skills_input="python scikit-learn pandas numpy"

if st.button("Predict Job Role"):
    if skills_input.strip() == "":
        st.warning("Please enter skills")
    else:
        vector = vectorizer.transform([skills_input.lower()])
        probs = model.predict_proba(vector)[0]
        classes = model.classes_
        sorted_idx = probs.argsort()[::-1]
        top_roles = classes[sorted_idx][:5]
        top_probs = probs[sorted_idx][:5]
        top_probs = top_probs / top_probs.sum()
        best_role = top_roles[0]
        confidence = round(top_probs[0]*100,2)
        st.session_state.best_role = best_role
        st.session_state.confidence = confidence
        st.session_state.skills_input = skills_input
        st.session_state.history.append({
            "skills": skills_input,
            "role": best_role,
            "confidence": confidence
        })
        st.success(f"🎯 Predicted Role: **{best_role}**")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={'suffix': "%"},
            title={'text': "Prediction Score"},
            gauge={'axis': {'range':[0,100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(
            x=top_roles,
            y=top_probs,
            labels={"x":"Role","y":"Confidence"},
            title="Top Career Matches",
            color=top_probs,
            color_continuous_scale="purples"
        )
        st.plotly_chart(fig2, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- SALARY INSIGHTS ---------------- #
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.header("💰 Salary Insights (Approx)")
salary = {
    "Data Scientist":"₹8L – ₹25L","Web Developer":"₹4L – ₹15L","DevOps Engineer":"₹10L – ₹30L",
    "Data Analyst":"₹4L – ₹12L","Backend Developer":"₹6L – ₹20L","Frontend Developer":"₹5L – ₹18L",
    "AI Engineer":"₹12L – ₹35L","Machine Learning Engineer":"₹10L – ₹30L","Cloud Engineer":"₹8L – ₹28L",
    "Database Administrator":"₹6L – ₹18L"
}
if st.button("Predict Salary"):
    if st.session_state.best_role:
        if st.session_state.best_role in salary:
            st.info(f"Average Salary Range for **{st.session_state.best_role}**: {salary[st.session_state.best_role]} per year")
        else:
            st.warning("Salary data not available for this role.")
    else:
        st.warning("Please predict your role first!")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- SUGGESTED SKILLS ---------------- #
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.header("🧠 Suggested Skills To Learn")
suggestions = {
    "Data Scientist":["Deep Learning","NLP","MLOps","Feature Engineering","Model Optimization","Data Pipelines"],
    "Web Developer":["Next.js","TypeScript","API Security","Web Performance","SEO","Progressive Web Apps"],
    "DevOps Engineer":["Terraform","Monitoring","Kubernetes Advanced","Infrastructure as Code","CI/CD Optimization","Cloud Security"],
    "Data Analyst":["Tableau","Statistics","Data Storytelling","Advanced Excel","Dashboard Design","Business Intelligence"],
    "Backend Developer":["System Design","Caching","Message Queues","Microservices","Authentication","API Scaling"],
    "Frontend Developer":["TypeScript","Next.js","State Management","Performance Optimization","UI/UX Design","Accessibility"],
    "AI Engineer":["Model Deployment","MLOps","Distributed Training","AI Optimization","Vector Databases","LLMs"],
    "Machine Learning Engineer":["Model Monitoring","Feature Stores","Pipeline Automation","Model Serving","Experiment Tracking"],
    "Cloud Engineer":["Infrastructure Automation","Cloud Security","Serverless Architecture","Networking","Cost Optimization"],
    "Database Administrator":["Query Optimization","Indexing","Replication","Backup Strategies","Database Scaling"]
}
if st.button("Suggest Skills"):
    if st.session_state.best_role in suggestions:
        st.session_state.recommended_skills = suggestions[st.session_state.best_role]
        for skill in st.session_state.recommended_skills:
            st.write("➡️", skill)
    else:
        st.warning("Please predict your role first!")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- LEARNING RESOURCES ---------------- #
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.header("📚 Learning Resources")
resources = {
    "Data Scientist":"https://www.kaggle.com/learn",
    "Web Developer":"https://developer.mozilla.org",
    "DevOps Engineer":"https://roadmap.sh/devops",
    "Data Analyst":"https://www.coursera.org",
    "Backend Developer":"https://www.udemy.com",
    "Frontend Developer":"https://www.freecodecamp.org",
    "AI Engineer":"https://www.coursera.org/learn/ai",
    "Machine Learning Engineer":"https://www.coursera.org/learn/machine-learning"
}
if st.session_state.best_role in resources:
    st.markdown(f"[Open Resource]({resources[st.session_state.best_role]})")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- AI CAREER REPORT ---------------- #
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.header("📥 AI Career Prediction Report")
st.markdown(f"**Predicted Role:** {st.session_state.best_role}")
st.markdown(f"**Confidence:** {st.session_state.confidence:.2f}%")
if st.session_state.recommended_skills:
    st.markdown("**Recommended Skills:** " + ", ".join(st.session_state.recommended_skills))
report = f"""
AI Career Prediction Report

Skills: {st.session_state.skills_input}
Predicted Role: {st.session_state.best_role}
Confidence: {st.session_state.confidence:.2f}%
Recommended Skills: {', '.join(st.session_state.recommended_skills) if st.session_state.recommended_skills else 'N/A'}

Generated by AI Job Role Predictor
"""
st.download_button(label="Download Report", data=report, file_name="career_report.txt")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- RESUME UPLOAD ---------------- #
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.header("📄 Resume Upload → Predict Role")
uploaded_file = st.file_uploader("Upload Resume (txt)", type=["txt"])
if uploaded_file:
    resume_text = uploaded_file.read().decode("utf-8")
    vector = vectorizer.transform([resume_text])
    prediction = model.predict(vector)[0]
    st.success(f"Predicted Role: **{prediction}**")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- SKILL GAP ---------------- #
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.header("🧠 Skill Gap Analyzer")
target_role = st.selectbox("Choose Target Role", list(resources.keys()))
required_skills = {
    "Data Scientist":["python","machine learning","statistics","pandas","numpy","data visualization"],
    "Web Developer":["html","css","javascript","react","git"],
    "DevOps Engineer":["aws","docker","kubernetes","ci/cd","linux"],
    "Data Analyst":["sql","excel","powerbi","python","data analysis"],
    "Backend Developer":["python","django","sql","api","database"],
    "Frontend Developer":["html","css","javascript","react","ui"],
    "AI Engineer":["python","tensorflow","deep learning","neural networks"],
    "Machine Learning Engineer":["python","scikit-learn","pandas","model deployment"],
    "Cloud Engineer":["aws","cloud","terraform","docker"],
    "Database Administrator":["sql","mysql","postgresql","database management"]
}
if st.button("Analyze Skill Gap"):
    if not st.session_state.best_role:
        st.warning("Predict a role first!")
    else:
        user_skills = st.session_state.skills_input.lower().split(",")
        missing = [skill for skill in required_skills.get(target_role, []) if skill not in user_skills]
        if missing:
            st.warning("Missing Skills:")
            st.write(missing)
        else:
            st.success("You have all required skills!")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- CAREER ROADMAP ---------------- #
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.header("🗺 AI Career Roadmap")
roadmaps = {
    "Data Scientist":["Python","Statistics","Pandas Numpy","Machine Learning","Deep Learning","Data Visualization","Projects"],
    "Web Developer":["HTML CSS","JavaScript","React","Backend","Database","API","Deploy"],
    "DevOps Engineer":["Linux","Docker","Kubernetes","CI/CD","Cloud","Monitoring","Automation"],
    "Data Analyst":["Excel","SQL","Python","PowerBI","Tableau","Data Visualization"],
    "Backend Developer":["Programming Language","Databases","APIs","Frameworks","Authentication","Deployment"],
    "Frontend Developer":["HTML","CSS","JavaScript","React","State Management","UI/UX","Deployment"],
    "AI Engineer":["Python","Machine Learning","Deep Learning","TensorFlow/PyTorch","Model Deployment","Projects"],
    "Machine Learning Engineer":["Python","Statistics","Scikit-learn","Model Training","MLOps","Deployment"],
    "Cloud Engineer":["Networking Basics","Linux","Cloud Platform","Containers","Infrastructure as Code","Monitoring"],
    "Database Administrator":["SQL","Database Design","MySQL/PostgreSQL","Backup Recovery","Performance Tuning"]
}
role_choice_roadmap = st.selectbox("Choose Career for Roadmap", list(roadmaps.keys()))
if st.button("Generate Roadmap"):
    for step in roadmaps[role_choice_roadmap]:
        st.write("➡️", step)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TECH STACK ---------------- #
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.header("🛠 Tech Stack")
t1,t2,t3 = st.columns(3)
t4,t5,t6 = st.columns(3)
for i,col in enumerate([t1,t2,t3,t4,t5,t6],1):
    with col:
        if i==1: st.write("Python","Scikit-learn","Pandas","NumPy")
        if i==2: st.write("TF-IDF NLP","Machine Learning","Data Processing","Feature Engineering")
        if i==3: st.write("Streamlit","Plotly","Interactive UI","Custom CSS")
        if i==4: st.write("Joblib","Model Serialization","Prediction Pipeline")
        if i==5: st.write("Data Visualization","Plotly Charts","Analytics Dashboard")
        if i==6: st.write("Git & GitHub","Project Deployment","Web App Development")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- SUPPORTED CAREER DOMAINS ---------------- #
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.header("🌐 Supported Career Domains")
domains = ["Web Development","Artificial Intelligence","Data Science","Cloud & DevOps",
           "Database Management","Mobile App Development","Cyber Security","Machine Learning",
           "Software Engineering","Game Development","Business Analytics","QA & Automation Testing"]
cols = st.columns(4)
for i,domain in enumerate(domains):
    with cols[i%4]:
        st.write(f"💻 {domain}")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PLATFORM STATS ---------------- #
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.header("📊 Platform Stats")
st.metric("Predictions Made", len(st.session_state.history))
st.metric("Available Roles", len(resources))
st.metric("Example Skills", len(example_skills))
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FAQ ---------------- #
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<div class="card" id="faq">', unsafe_allow_html=True)
st.header("❓ FAQ")
st.write("Q: Do I need to enter all my skills? A: No, just key skills are enough.")
st.write("Q: Can I upload my resume? A: Yes, txt format only.")
st.write("Q: Can I download my report? A: Yes, click 'Download Report'.")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TESTIMONIALS ---------------- #
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<div class="card" id="testimonials">', unsafe_allow_html=True)
st.header("💬 Testimonials")
st.write("⭐ Kajal Siwach: 'This tool helped me discover the perfect career path!'")
st.write("⭐ AI Enthusiast: 'Accurate and interactive predictions!'")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- CONTACT ---------------- #
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<div class="card" id="contact">', unsafe_allow_html=True)
st.header("📩 Contact")
st.write("Email: kajalsiwach251@example.com")
st.write("LinkedIn: https://www.linkedin.com/in/kajal-siwach-8b971731a/")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ---------------- # 
st.markdown("""<center> Made with ❤️ by **Kajal Siwach** AI Portfolio Project | 2026 </center> """, unsafe_allow_html=True)