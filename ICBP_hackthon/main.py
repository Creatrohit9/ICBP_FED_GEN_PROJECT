import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import PyPDF2
import seaborn as sns
import matplotlib.pyplot as plt
import os
from googlesearch import search
from youtubesearchpython import VideosSearch
import requests
from bs4 import BeautifulSoup
import cohere
from wordcloud import WordCloud
import ollama
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import utils
import pyttsx3
import speech_recognition as sr
import pickle
import numpy as np

#load model for Student placment prediction
model = pickle.load(open('model.pkl', 'rb'))
    
    
# data: Program skills
program_data = {
    "Data Science": ["Python", "Machine Learning", "SQL", "Statistics", "Data Analysis"],
    "Web Development": ["HTML", "CSS", "JavaScript", "React", "Node.js"],
    "AI/ML Engineering": ["Python", "TensorFlow", "Deep Learning", "Neural Networks", "Computer Vision"],
    "Software Engineering": ["C++", "Java", "Algorithms", "OOP", "Data Structures"]
}

# Mapping programs to courses
program_courses = {
    "Data Science": ["Code Unnati", "Skill 4 Future"],
    "AI/ML Engineering": ["Code Unnati", "Skill 4 Future"],
    "Software Engineering": ["Tech Saksham"],
    "Web Development": ["Tech Saksham"]
}

# Convert skill lists into a string format for vectorization
program_skills = {program: " ".join(skills) for program, skills in program_data.items()}

# Vectorize the skills data
vectorizer = CountVectorizer()
program_vectors = vectorizer.fit_transform(program_skills.values())

# Function to suggest the best program based on user input
def suggest_program(user_skills):
    user_vector = vectorizer.transform([user_skills])
    similarity_scores = cosine_similarity(user_vector, program_vectors)
    
    # Get the program with the highest similarity score
    best_match_idx = similarity_scores.argmax()
    best_program = list(program_skills.keys())[best_match_idx]
    
    return best_program

# Function to suggest the best program based on user input
def suggest_program(user_skills):
    user_vector = vectorizer.transform([user_skills])
    similarity_scores = cosine_similarity(user_vector, program_vectors)
    
    # Get the program with the highest similarity score
    best_match_idx = similarity_scores.argmax()
    best_program = list(program_skills.keys())[best_match_idx]
    
    return best_program

# extract text from pdf
def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    # Iterate over all pages and extract text
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text() + "\n"
    return text

# Diffrent Roles for resume match
roles = {
    'Android Developer': ['java', 'kotlin', 'android', 'sdk','gradle', 'mvvm', 'database', 'firebase', 'sqlite'],
    'Full Stack Developer': ['javascript', 'react', 'node', 'css', 'html', 'sql', 'redux', 'mongodb', 'express', 'typescript', 'next.js', 'graphql', 'docker', 'kubernetes', 'restful api', 'jwt', 'oauth2', 'jest', 'webpack', 'sass', 'heroku', 'aws lambda', 'cloudflare'],
    'Backend Developer': ['django', 'flask', 'sql', 'aws', 'redis', 'kubernetes','nginx', 'gunicorn', 'graphql', 'restful', 'api','mongodb', 'postgresql', 'unit testing', 'jenkins', 'cloudformation'],
    'Python Developer': [ 'python', 'pandas', 'numpy', 'flask', 'django', 'fastapi', 'pytest', 'sqlalchemy', 'matplotlib', 'dash', 'tkinter', 'openpyxl', 'xml parsing', 'beautifulsoup', 'jupyter', 'dataclasses', 'docker', 'uvicorn','sockets'],
    'Machine Learning Engineer': ['python', 'tensorflow', 'keras', 'pandas', 'scikit-learn', 'numpy', 'matplotlib', 'seaborn', 'pytorch', 'xgboost', 'lightgbm', 'statsmodels', 'nltk', 'spacy', 'gensim', 'opencv', 'huggingface', 'transformers','feature engineering', 'deployment','aws sagemaker', 'azure ml', 'data preprocessing'],
    'Graphic Designer': [ 'photoshop', 'illustrator', 'adobe', 'creativity', 'design', 'indesign', 'xd', 'sketch', 'figma', 'coreldraw', 'typography', 'color theory', 'branding', 'motion graphics', 'after effects', 'lightroom', 'premiere pro', 'ux/ui design', 'blender', '3d modeling', 'infographics', 'visual storytelling'],
}


# Function to match skills to job roles
def recommend_role_based_on_skills(extracted_text):
    
    recommendations = []
    extracted_text = extracted_text.lower()
    
    # Check for skill matches
    for role, skills in roles.items():
        skill_count = sum(1 for skill in skills if skill in extracted_text)
        if skill_count > 0:
            recommendations.append((role, skill_count))
    
    # Sort recommendations by the number of matching skills
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations


# Function to recommend roles based on skills in the extracted resume text
def recommend_role_based_on_skills(extracted_text):
    recommendations = []
    extracted_text = extracted_text.lower()
    
    # Check for skill matches
    for role, skills in roles.items():
        skill_count = sum(1 for skill in skills if skill in extracted_text)
        if skill_count > 0:
            recommendations.append((role, skill_count, len(skills)))
    
    # Sort recommendations by the number of matching skills
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:3]  # Return top 3 recommendations

# Function to create a single donut chart for the top 3 roles
def plot_donut_chart(recommendations):
    # Prepare data for the donut chart
    labels = [f"{role}" for role, matched_skills, total_skills in recommendations]
    #labels = [f"{role} ({(matched_skills/total_skills) * 100:.1f}%)" for role, matched_skills, total_skills in recommendations]
    sizes = [(matched_skills / total_skills) * 100 for _, matched_skills, total_skills in recommendations]

    # Plot the donut chart
    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, _ = ax.pie(
        sizes, labels=labels, autopct='%.1f%%', startangle=90, colors=sns.color_palette('pastel')  )

    # Add a circle at the center to create the "donut" effect
    center_circle = plt.Circle((0, 0), 0.60, fc='white')
    fig.gca().add_artist(center_circle)

    # Display the chart
    st.pyplot(fig)
    
def generate_wordcloud(resume_text):
    # Generate the word cloud using the provided resume text
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(resume_text)
    
    # Display the word cloud using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Hide the axes
    st.pyplot(plt)
    
# Function to fetch top 5 Google search results
def get_google_links(query):
    search_results = search(query, num_results=5)
    return search_results


# Initialize the Cohere client (replace with your actual API key)
cohere_client = cohere.Client("N1eiSCuo6f75v4ZW3hobIGx5ivLV1zSbOETTw3bB")

# Function to generate interview questions for a given role
def generate_questions(role):
    prompt = f"Generate a list of interview questions for a {role} position."
    response = cohere_client.generate(
        model="command-r-plus-08-2024",  # You can choose different models
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
    )
    
    # Split the result into individual questions based on newlines
    return response.generations[0].text.split('\n')

# Function to fetch top 5 YouTube video links
def get_youtube_videos(query):
    # Perform YouTube search with the query
    videos_search = VideosSearch(query + " learning resources", limit=5)
    try:
        videos_result = videos_search.result()
        
        # Log the response to see the exact structure
        #print("Full Response:", videos_result)
        
        # Check if there are any results in the response
        if 'result' in videos_result and len(videos_result['result']) > 0:
            # Extract the video links
            video_links = [video.get('link') for video in videos_result['result']]
        else:
            video_links = []
            print("No videos found based on your query.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        video_links = []
    
    return video_links


# Function to display relevant PDFs and CSV files based on selected role
def display_learning_resources(selected_role):
    # Path to learning resources folder
    learning_folder = "learning_resources"  # Replace with the path to your folder
    
    # Create a list of available PDFs and CSVs for the selected role
    resources = []
    
    # Check if the learning folder exists
    if os.path.exists(learning_folder):
        # List all files in the folder
        files = os.listdir(learning_folder)
        
        # Filter PDFs and CSVs based on the selected role
        for file in files:
            if file.lower().startswith(selected_role.lower()):
                if file.endswith(".pdf"):
                    resources.append(file)
        
        # Provide download options for each relevant resource
        for resource in resources:
            resource_path = os.path.join(learning_folder, resource)
            st.download_button(
                label=f"Download {resource}",
                data=open(resource_path, "rb").read(),
                file_name=resource,
                mime="application/octet-stream"
            )
            
                       
# Function to fetch the content from the URL
def fetch_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extracting text content from paragraphs
    paragraphs = soup.find_all('p')
    content = " ".join([para.get_text() for para in paragraphs])
    return content

# Function to display course details based on the selected option
def show_course_details(course):
            if course == "Foundation Course":
                st.subheader("II Year - Foundation Course")
                st.write("""
                **Overview of Foundation Course:**
                The Foundation Course under Code Unnati Program will be offered to the second and pre-final students pursuing engineering and other technical degree courses. 
                This course will cover the pre-requisites required for Artificial Intelligence, Machine Learning, Data Analytics, Deep Learnings, and Computer Vision Technologies of the Code Unnati Advance and Value-Added Course.

                **Learning Outcomes:**
                - Demonstrate fundamentals of Python tools and its data analytics libraries.
                - Able to generate visualizations using Python.
                - Understand the concept and applications of AI.
                - Understanding industry-specific SAP tools.

                **Foundation Course Outline (50 Hours):**
                1. Python Programming Language
                2. Data Analysis with Python
                3. Artificial Intelligence
                4. SAP Conversational AI Chatbot
                5. Capstone Project
                """)
            
            elif course == "Advance Course":
                st.subheader("III Year - Advance Course")
                st.write("""
                **Overview of Advance Course:**
                Code Unnati for the Advance course will be offered to the pre-final/final year students pursuing engineering and other technical degree courses. 
                This course will cover the Advanced Concepts of AI, Machine Learning, Data Analytics, Deep Learning, Computer Vision with OpenVINO Toolkit, and IoT with hands-on learning.

                **Learning Outcomes:**
                - Apply the basic principles, models, and algorithms of AI.
                - Analyze structures and algorithms related to AI and machine learning.
                - Design and implement machine learning algorithms in real-world applications.
                - Solve real-life challenges using Computer Vision technology.
                - Design Prototype level solutions using IoT.
                - Demonstrate SAP Technical Modules such as ABAP on the Business Technology Platform (BTP).

                **Advance Course Outline (70 Hours):**
                1. Foundation Crash Course
                2. Statistical Modelling and Predictive Analytics – Machine Learning
                3. Internet of Things
                4. Deep Learning and Computer Vision OpenVINO Toolkit
                5. SAP ABAP on Business Technology Platform (BTP)
                6. Capstone Project
                """)
            
            elif course == "Value-Added Course":
                st.subheader("IV Year - Value-Added Course")
                st.write("""
                **Overview of Value-Added Course:**
                Code Unnati for the Value-Added course will be offered to students who have completed the Advance Course of Code Unnati. 
                This course will cover IoT Cybersecurity and SAP Analytics Cloud with practical hands-on learning.

                **Learning Outcomes:**
                - Application of security measures in IoT.
                - Demonstrate SAP Analytics Cloud.

                **Value-Added Course Outline (40 Hours):**
                1. IoT Cybersecurity
                2. SAP Analytics Cloud
                3. Capstone Project
                """)
                
# Function to plot donut chart for individual students
def plot_donut_chart1(student_row):
    labels = ['Mid Assessment', 'Final Assessment']
    sizes = [student_row['Mid'], student_row['Score']]
    
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'], pctdistance=0.85)
    
    # Draw inner circle for donut effect
    center_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(center_circle)
    
    ax.set_title(f"Performance of {student_row['Name']}")
    plt.axis('equal')
    plt.tight_layout()
    
    return fig
    
    
# Sidebar for navigation
def main():
    st.set_page_config(page_title='AlphaTeam', layout='wide',)

    with st.sidebar:
        selected = option_menu('GENAI & Federated Learning',
                               ['Edunet Foundation','Program', 'Code Unnati', 'Placement','Placement Prediction'],
                               icons=['person','activity','pie-chart', 'bar-chart','book'],
                               default_index=0)


        # Set the logo and title for each menu item
        if selected == 'Edunet Foundation':
            logo = Image.open('images/edunet_logo.png')
            title = "Edunet Foundation"
        elif selected == 'Program':
            logo = Image.open('images/Program_logo.png')
            title = "Program"
        elif selected == 'Code Unnati':
            logo = Image.open('images/code_unnati_logo.png')
            title = "Code Unnati"
        elif selected == 'Placement':
            logo = Image.open('images/placement_bot_logo.png')
            title = "Placement"
        elif selected == 'Placement Prediction':
            logo = Image.open('images/placement_bot_logo.png')
            title = "Placement Prediction"
    
        # Display the logo in the sidebar
        st.image(logo, width=200)
        st.title(title)

    # Page content based on the selected option
    if selected == 'Edunet Foundation':

                        
        # Load bot logo image
        bot_logo = Image.open("images/robot.png")

        # Initialize chat history in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Create a two-column layout for bot logo and chat
        col1, col2 = st.columns([1, 3])  # Adjust column width as needed

        with col1:
            # Display bot logo
            st.image(bot_logo, caption="Edunet Bot", use_container_width=True)

        with col2:
            st.title("Chat with Edunet Bot")

            # Display chat history
            for sender, message in st.session_state.chat_history:
                if sender == "You":
                    # User messages aligned right
                    st.markdown(f"<div style='text-align:right;'><strong>{sender}:</strong> {message}</div>", unsafe_allow_html=True)
                else:
                    # Bot messages aligned left with styling
                    st.markdown(f"<div style='text-align:left; background-color:#f1f1f1; padding:10px; border-radius:10px;'>"
                                f"<strong>{sender}:</strong> {message}</div>", unsafe_allow_html=True)

            st.divider()

            # Text input for prompt
            prompt = st.text_input("Ask anything about Edunet Foundation...", placeholder="Enter your question here...", key="url_prompt")

            # Submit button
            submit_btn = st.button("Submit")

            # Button click logic
            if submit_btn and prompt:
                # Directly append user's input to chat history and render the response
                st.session_state.chat_history.append(("You", prompt))
                
                # Show spinner while processing the response
                with st.spinner("Processing..."):
                    response = utils.ask_gemini(prompt)  # Your bot API call
                
                # Append the bot's response to the chat history
                st.session_state.chat_history.append(("Bot", response))

                # To force a rerun without using `experimental_rerun`, just trigger a small change in session state
                st.session_state['last_interaction'] = prompt  # Dummy state change to trigger a rerun

            st.divider()
            st.caption("Powered by @Alpha Team Edunet Foundation")
                
                    
                    
    elif selected == "Program":
        # Title of the app with a modern style
        st.markdown("<h1 style='text-align: left; color: #ADD8E6;'>Program Suggestion AI</h1>", unsafe_allow_html=True)

        # Display a short description
        st.markdown("""
            <p style="text-align: left; font-size: 16px; color: #666;">Find the best program and related courses based on your skills.</p>
        """, unsafe_allow_html=True)

        # Display a divider for clean separation
        st.markdown("---")

        # Display input widget for skills selection
        st.subheader("Select your skills:")
        all_skills = sorted(set(skill for skills in program_data.values() for skill in skills))

        # Using a multiselect widget for skill selection
        selected_skills = st.multiselect("Choose your skills:", all_skills, help="Hold 'Ctrl' to select multiple skills")

        # Add a button for generating suggestions
        if st.button("Suggest Best Program and Courses", key="suggest_button", help="Click to get recommendations based on your selected skills"):
            if selected_skills:
                # Show a loading spinner while processing the request
                with st.spinner('Finding the best program...'):
                    user_skills = " ".join(selected_skills)  # Convert skills to a space-separated string
                    recommended_program = suggest_program(user_skills)
                    
                    # Simulate fetching courses for the recommended program
                    suggested_courses = program_courses.get(recommended_program, [])
                    
                    # Display the result in a more stylish manner
                    st.markdown(f"### 🎯 **The Best Program for You:** {recommended_program}")
                    st.markdown("Here are the recommended courses you can explore:")
                    for course in suggested_courses:
                        st.markdown(f"- **{course}**")
                        
                    st.success("Your program and courses have been successfully recommended!")
            else:
                st.warning("Please select at least one skill to proceed.")
        st.divider()
        # Add some footer text to enhance the experience
        st.caption("Powered by @Alpha Team Edunet Foundation")
    elif selected == "Code Unnati":
                    
        # Header with logo
        st.markdown("""
            <style>
                .header {
                    background-color: #004A94;
                    color: white;
                    padding: 15px;
                    text-align: center;
                    font-size: 24px;
                    font-weight: bold;
                }
                .section-header {
                    color: #004A94;
                    font-size: 20px;
                    font-weight: bold;
                    margin-top: 20px;
                }
                .section-summary {
                    color: #333;
                    font-size: 16px;
                    margin-bottom: 20px;
                }
                .table-style {
                    background-color: #f4f4f9;
                    border-radius: 10px;
                }
            </style>
            <div class="header">Code Unnati Initiative</div>
        """, unsafe_allow_html=True)

        # Display logo
        image = Image.open('images/code_unn.jpg') 
        st.image(image, caption='Code Unnati Initiative', use_container_width=True)

        # Code Unnati Summary Section
        st.subheader("Code Unnati Initiative")
        st.markdown("""
            <div class="section-summary">
                **Code Unnati** is a CSR initiative of **SAP India** aimed at enhancing the skills and employability of youth in **Industry 4.0**. 
                The initiative provides advanced tech infrastructure and offers training on emerging technologies like **Artificial Intelligence, Machine Learning, Cloud Computing, Computer Vision, IoT, SAP BTP**, and **ABAP**. 
            </div>
        """, unsafe_allow_html=True)

        # Create table for hours and courses
        data = {
            "Content Type": [
                "Core Deep Tech Offering", 
                "Industry Specific Modular Offering", 
                "Employability Skills", 
                "Capstone Project"
            ],
            "2nd Year (Hours)": ["~40", "~10", "-", "~20"],
            "3rd Year (Hours)": ["~55", "~15", "~15", "~30"],
            "4th Year (Hours)": ["~20", "~20", "~15", "~60"],
            "Delivery Type": ["Instructor-led", "Instructor-led", "Instructor-led", "Instructor-led"],
            "Delivery Mode": ["Hybrid", "Hybrid", "Hybrid", "Hybrid"]
        }
        df = pd.DataFrame(data)
        st.table(df)

        # Display total hours for each year
        total_hours = {
            "Year": ["2nd Year", "3rd Year", "4th Year"],
            "Total Hours": [70, 115, 115]
        }
        df_total_hours = pd.DataFrame(total_hours)
        st.subheader("Year-wise Total Hours")
        st.table(df_total_hours)

        # Interactive course selection
        course_option = st.selectbox("Select a course", ["Foundation Course", "Advance Course", "Value-Added Course"])

        # Function to show course details
        def show_course_details(course_option):
            st.write(f"Showing details for **{course_option}**...")

        show_course_details(course_option)

        # Student performance section
        st.title("Student Performance Assessment")
        st.write("""
            Upload a CSV or Excel file containing student performance data, and the app will visualize:
            - Top 5 performers in both mid and final assessments
            - Average scores
            - Individual donut charts comparing mid and final assessments
            - Placement comparison according to high scores and skills.
        """)

        # File upload section
        uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])

        # Process uploaded file
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                student_data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                student_data = pd.read_excel(uploaded_file)
            
            if 'Name' in student_data.columns and 'Mid' in student_data.columns and 'Score' in student_data.columns:
                avg_mid = student_data['Mid'].mean()
                avg_final = student_data['Score'].mean()
                st.write(f"Average Mid Assessment Score: {avg_mid:.2f}")
                st.write(f"Average Final Assessment Score: {avg_final:.2f}")

                top_5_final = student_data.nlargest(5, 'Score')
                top_5_mid = student_data.nlargest(5, 'Mid')

                # Plot top performers in a side-by-side layout
                st.subheader("Top 5 Performers in Final and Mid Assessments")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                sns.barplot(x='Score', y='Name', data=top_5_final, ax=ax1, palette='Blues_d')
                ax1.set_title('Top 5 Final Assessment Scores')
                ax1.set_ylabel('Name')
                ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

                sns.barplot(x='Mid', y='Name', data=top_5_mid, ax=ax2, palette='Greens_d')
                ax2.set_title('Top 5 Mid Assessment Scores')
                ax2.set_ylabel('Mid Score')
                ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

                st.pyplot(fig)

                # Donut charts section
                st.subheader("Individual Student Performance (Donut Charts)")
                student_names = st.multiselect("Select students to view", options=student_data['Name'].unique())
                
                if student_names:
                    for student in student_names:
                        student_row = student_data[student_data['Name'] == student].iloc[0]
                        fig = plot_donut_chart1(student_row)  # You need to define this function to create donut charts
                        st.pyplot(fig)

                # Comparison bar chart
                st.subheader("All Students' Mid and Final Assessment Scores")
                student_data_melted = student_data.melt(id_vars=["Name"], value_vars=["Mid", "Score"], 
                                                        var_name="Assessment", value_name="Score Value")
                fig, ax = plt.subplots(figsize=(15, 8))
                sns.barplot(x="Name", y="Score Value", hue="Assessment", data=student_data_melted, palette="Set2", ax=ax)
                ax.set_title('Mid vs Final Assessment Scores for All Students')
                ax.set_ylabel('Score')
                ax.set_xlabel('Student Name')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)

            else:
                st.error("The uploaded file must contain 'Name', 'Mid', and 'Score' columns.")
        else:
            st.info("Please upload a CSV or Excel file to proceed.")

        # Add divider
        st.divider()

        # Prompt for external interaction
        prompt = st.text_input("Go Interact", placeholder="What is the Edunet Foundation?", key="url_prompt")
        submit_btn = st.button(label="Submit", key="url_btn")

        if submit_btn:
            with st.spinner("Processing..."):
                response = utils.ask_gemini(prompt)
                st.markdown(response)
                st.divider()
        st.divider()    
        st.caption("Powered by @Alpha Team Edunet Foundation")
        
                
    elif selected =='Placement Prediction':
        # Add title and description
        st.title("Student Placement Prediction")
        st.write("Enter your details below to predict your placement chances")

        # Create input fields
        with st.form("prediction_form"):
            # Gender selection
            gender = st.selectbox(
                "Select Gender",
                options=["Male", "Female"],
                help="Select your gender"
            )
            gender = 1 if gender == "Male" else 0

            # Stream selection
            stream = st.selectbox(
                "Select Stream",
                options=["Computer Science", "Information Technology", "Electronics", "Mechanical"],
                help="Select your stream"
            )
            # Convert stream to encoded value (assuming same encoding as original)
            stream_mapping = {
                "Computer Science": 0,
                "Information Technology": 1,
                "Electronics": 2,
                "Mechanical": 3
            }
            stream = stream_mapping[stream]

            # Internship
            internship = st.radio(
                "Have you done any internship?",
                options=["Yes", "No"],
                help="Select whether you have completed any internship"
            )
            internship = 1 if internship == "Yes" else 0

            # CGPA
            cgpa = st.slider(
                "Your CGPA",
                min_value=0.0,
                max_value=10.0,
                value=7.0,
                step=0.1,
                help="Enter your CGPA"
            )

            # Backlogs
            backlogs = st.number_input(
                "Number of Backlogs",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
                help="Enter number of backlogs"
            )

            # Submit button
            submit = st.form_submit_button("Predict Placement Chances")

        # Make prediction when form is submitted
        if submit:
            # Create input array for prediction
            input_data = np.array([gender, stream, internship, cgpa, backlogs])
            input_data = input_data.astype(float)
            
            # Get prediction
            prediction = model.predict([input_data])[0]
            
            # Show prediction with custom styling
            if prediction == 1:
                st.success("🎉 You have high chances of getting placed!")
                st.balloons()
            else:
                st.warning("📚 You have lower chances of getting placed. Keep working hard!")
                
            # Show input summary
            st.subheader("Your Input Summary:")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Gender:", "Male" if gender == 1 else "Female")
                st.write("Stream:", list(stream_mapping.keys())[list(stream_mapping.values()).index(stream)])
                st.write("Internship:", "Yes" if internship == 1 else "No")
            with col2:
                st.write("CGPA:", cgpa)
                st.write("Backlogs:", int(backlogs))

        st.divider()    
        st.caption("Powered by @Alpha Team Edunet Foundation")
        
                            
    elif selected == 'Placement':
        st.header("Placement Empower")
        st.write("Please upload your resume (PDF) to get skillset recommendations for job roles.")
        # File uploader for resume
        resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
        
        if resume_file is not None:
            # Extract text from the uploaded PDF file
            with st.spinner("Extracting text from your resume..."):
                resume_text = extract_text_from_pdf(resume_file)
                st.subheader("Generated Word Cloud:")
                generate_wordcloud(resume_text)
            if resume_text:
                #st.subheader("Resume Text")
                #st.write(resume_text)
                # Recommend roles based on the extracted resume text
                st.subheader("Recommended Job Roles Based on Your Skills")
                recommendations = recommend_role_based_on_skills(resume_text)
        
                if recommendations:
                    st.write("Top 3 Recommended Roles:")
                    for role, matched_skills, total_skills in recommendations:
                        st.write(f"{role}")

                    # Plot donut chart for the top 3 roles
                    plot_donut_chart(recommendations)
                    
                    
                    job_action = st.radio("What would you like to do next?", ("Start Preparing", "Give Interview"))
        
                    if job_action == "Start Preparing":
                        selected_role = st.selectbox("Select a role to start preparing", [role for role, _, _ in recommendations])

                        # Display relevant learning resources (PDFs, CSVs) based on the selected role
                        st.write(f"Here are the learning resources for {selected_role}:")
                        display_learning_resources(selected_role)
                        
                        # Fetch top 5 Google links for the selected role
                        google_links = get_google_links(selected_role + " learning resources")
                        st.write("Top 5 Google links for learning:")
                        for link in google_links:
                            st.markdown(f"[{link}]({link})")
                        
                        # Fetch top 5 YouTube videos for the selected role
                        youtube_links = get_youtube_videos(selected_role + " learning resources")
                        st.write("Top 5 YouTube videos for learning:")
                        for link in youtube_links:
                            st.markdown(f"[Watch Video - {link}]({link})")
                            
                            
                    elif job_action == "Give Interview":
                        selected_role = st.selectbox("Select a role to start preparing", [role for role, _, _ in recommendations])

                        st.write("You selected 'Give Interview'. Here are some interview tips:")
                        # Provide interview tips based on the selected role
                        questions = generate_questions(selected_role)
                    
                        # Display the generated questions
                        if questions:
                            st.subheader(f"Interview Questions for {selected_role}:")
                            for i, question in enumerate(questions, 1):
                                st.write(f"{question.strip()}")
                        else:
                            st.error("No questions generated. Try again.")
                    
                else:
                    st.write("No matching roles found based on the resume.")
            else:
                st.write("Could not extract text from the resume. Please try again.")
        st.divider()    
        st.caption("Powered by @Alpha Team Edunet Foundation")
        
if __name__ == "__main__":
    main()
