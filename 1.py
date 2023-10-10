import random
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


# Load the IMDb dataset from the CSV file
@st.cache_data
def load_imdb_data():
    imdb_data = pd.read_csv(r'C:\Users\user\Downloads\IMDB Dataset.csv')
    return imdb_data


# Load the second IMDb dataset
@st.cache_data
def load_second_imdb_data():
    second_imdb_data = pd.read_csv(r'C:\Users\user\Downloads\chat.csv')
    return second_imdb_data


imdb_data = load_imdb_data()
second_imdb_data = load_second_imdb_data()

# Extract the reviews and labels (positive or negative)
reviews = imdb_data['review'].tolist()
labels = imdb_data['sentiment'].tolist()

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform the training data
training_data_vectorized= vectorizer.fit_transform(reviews)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(training_data_vectorized, labels, test_size=0.2, random_state=42)

# Create the Multinomial Naive Bayes model
model = MultinomialNB()

# Train the model on the training data
model.fit(X_train, [1 if label == "positive" else 0 for label in y_train])

# Create a Streamlit web app
st.title("Movie Review Sentiment Analysis Chatbot")

# Create a function to generate a response
def generate_response(user_input):
    user_input_lower = user_input.lower()

    if user_input_lower.startswith("what is the user rating for"):
        movie_name = user_input[27:].strip()  # Extract the movie name from the user input
        movie_data = second_imdb_data[second_imdb_data['movie title'].str.lower() == movie_name.lower()]
        if not movie_data.empty:
            user_rating = movie_data.iloc[0]['User Rating']
            return f"The user rating for '{movie_name}' is {user_rating}."
        else:
            return f"Sorry, I couldn't find user rating information for '{movie_name}'."

    elif user_input_lower.startswith("what is the rating for"):
        movie_name = user_input[22:].strip()  # Extract the movie name from the user input
        movie_data = second_imdb_data[second_imdb_data['movie title'].str.lower() == movie_name.lower()]
        if not movie_data.empty:
            rating = movie_data.iloc[0]['Rating']
            return f"The rating for '{movie_name}' is {rating}."
        else:
            return f"Sorry, I couldn't find rating information for '{movie_name}'."

    elif user_input_lower.startswith("what is the genre for"):
        movie_name = user_input[21:].strip()  # Extract the movie name from the user input
        movie_data = second_imdb_data[second_imdb_data['movie title'].str.lower() == movie_name.lower()]
        if not movie_data.empty:
            genre = movie_data.iloc[0]['Generes']
            return f"The genre for '{movie_name}' is {genre}."
        else:
            return f"Sorry, I couldn't find genre information for '{movie_name}'."

    elif user_input_lower.startswith("what is the plot for"):
        movie_name = user_input[20:].strip()  # Extract the movie name from the user input
        movie_data = second_imdb_data[second_imdb_data['movie title'].str.lower() == movie_name.lower()]
        if not movie_data.empty:
            plot = movie_data.iloc[0]['Overview']
            return f"The plot for '{movie_name}' is {plot}."
        else:
            return f"Sorry, I couldn't find plot information for '{movie_name}'."

    elif user_input_lower.startswith("what is the overview for"):
        movie_name = user_input[24:].strip()  # Extract the movie name from the user input
        movie_data = second_imdb_data[second_imdb_data['movie title'].str.lower() == movie_name.lower()]
        if not movie_data.empty:
            overview = movie_data.iloc[0]['Overview']
            return f"The genre for '{movie_name}' is {overview}."
        else:
            return f"Sorry, I couldn't find overview information for '{movie_name}'."

    elif user_input_lower.startswith("who is the director for"):
        movie_name = user_input[23:].strip()  # Extract the movie name from the user input
        movie_data = second_imdb_data[second_imdb_data['movie title'].str.lower() == movie_name.lower()]
        if not movie_data.empty:
            director = movie_data.iloc[0]['Director']
            return f"The director for '{movie_name}' is {director}."
        else:
            return f"Sorry, I couldn't find director information for '{movie_name}'."

    elif user_input_lower.startswith("who is the writer for"):
        movie_name = user_input[21:].strip()  # Extract the movie name from the user input
        movie_data = second_imdb_data[second_imdb_data['movie title'].str.lower() == movie_name.lower()]
        if not movie_data.empty:
            writer = movie_data.iloc[0]['Writer']
            return f"The writer for '{movie_name}' is {writer}."
        else:
            return f"Sorry, I couldn't find writer information for '{movie_name}'."

    elif user_input_lower.startswith("who is the cast for"):
        movie_name = user_input[19:].strip()  # Extract the movie name from the user input
        movie_data = second_imdb_data[second_imdb_data['movie title'].str.lower() == movie_name.lower()]
        if not movie_data.empty:
            cast = movie_data.iloc[0]['Top 5 Casts']
            return f"The cast for '{movie_name}' is {cast}."
        else:
            return f"Sorry, I couldn't find cast information for '{movie_name}'."

    elif user_input_lower.startswith("when is the year of release for"):

        movie_name = user_input_lower[31:].strip()  # Extract the movie name from the user input

        movie_data = second_imdb_data[second_imdb_data['movie title'].str.lower() == movie_name]

        if not movie_data.empty:

            release_year = movie_data.iloc[0]['year']

            return f"The year of release for '{movie_name}' is {release_year}."

        else:

            return f"Sorry, I couldn't find release year information for '{movie_name}'."

    elif user_input_lower.startswith("tell me about"):
        movie_name = user_input_lower[13:].strip()  # Extract the movie name from the user input
        movie_data = second_imdb_data[second_imdb_data['movie title'].str.lower() == movie_name]
        if not movie_data.empty:
            response = f"Here are the details for '{movie_name}':\n"
            response += f"Rating: {movie_data.iloc[0]['Rating']}\n"
            response += f"Genres: {movie_data.iloc[0]['Generes']}\n"
            response += f"Overview: {movie_data.iloc[0]['Overview']}\n"
            response += f"Plot Keyword: {movie_data.iloc[0]['Plot Kyeword']}\n"
            response += f"Director: {movie_data.iloc[0]['Director']}\n"
            response += f"Top 5 Casts: {movie_data.iloc[0]['Top 5 Casts']}\n"
            response += f"Writer: {movie_data.iloc[0]['Writer']}\n"
            response += f"Year: {movie_data.iloc[0]['year']}\n"
            return response
        else:
            return f"Sorry, I couldn't find information for '{movie_name}'."

    elif user_input_lower.startswith("what do you think about"):
        movie_name = user_input_lower[23:].strip()  # Extract the movie name from the user input
        movie_data = second_imdb_data[second_imdb_data['movie title'].str.lower() == movie_name]
        if not movie_data.empty:
            response = f"Here are the details for '{movie_name}':\n"
            response += f"Rating: {movie_data.iloc[0]['Rating']}\n"
            response += f"Genres: {movie_data.iloc[0]['Generes']}\n"
            response += f"Overview: {movie_data.iloc[0]['Overview']}\n"
            response += f"Plot Keyword: {movie_data.iloc[0]['Plot Kyeword']}\n"
            response += f"Director: {movie_data.iloc[0]['Director']}\n"
            response += f"Top 5 Casts: {movie_data.iloc[0]['Top 5 Casts']}\n"
            response += f"Writer: {movie_data.iloc[0]['Writer']}\n"
            response += f"Year: {movie_data.iloc[0]['year']}\n"
            return response
        else:
            return f"Sorry, I couldn't find information for '{movie_name}'."

    elif user_input_lower.startswith("what do you think of"):
        movie_name = user_input_lower[20:].strip()  # Extract the movie name from the user input
        movie_data = second_imdb_data[second_imdb_data['movie title'].str.lower() == movie_name]
        if not movie_data.empty:
            response = f"Here are the details for '{movie_name}':\n"
            response += f"Rating: {movie_data.iloc[0]['Rating']}\n"
            response += f"Genres: {movie_data.iloc[0]['Generes']}\n"
            response += f"Overview: {movie_data.iloc[0]['Overview']}\n"
            response += f"Plot Keyword: {movie_data.iloc[0]['Plot Kyeword']}\n"
            response += f"Director: {movie_data.iloc[0]['Director']}\n"
            response += f"Top 5 Casts: {movie_data.iloc[0]['Top 5 Casts']}\n"
            response += f"Writer: {movie_data.iloc[0]['Writer']}\n"
            response += f"Year: {movie_data.iloc[0]['year']}\n"
            return response
        else:
            return f"Sorry, I couldn't find information for '{movie_name}'."

    elif user_input_lower.startswith("what is the review for"):
         movie_name = user_input_lower[21:].strip()
         movie_data = imdb_data[imdb_data['review'].str.lower() == movie_name]
         if not movie_data.empty:
             response = f"Here is the sentiment for '{movie_name}':\n"
             response +=f"Sentiment: {movie_data.iloc[0]['Sentiment']}\n"

    else:
        # Vectorize the user input
        user_input_vectorized = vectorizer.transform([user_input])

        # Predict the class of the user input
        class_prediction = model.predict(user_input_vectorized)[0]

        # Define possible responses
        positive_responses = [
            "I'm glad you liked the movie!",
            "That's great to hear!",
            "That's awesome!",
            "That's mind-blowing. I hope the next one lives up to the same level of hype as this one!"
            "Sounds like a fantastic movie experience!",
            "Awesome! Your enthusiasm for the movie is infectious.",
            "It's always wonderful to hear positive reviews!",
        ]

        negative_responses = [
            "I'm sorry you did not like the movie.",
            "It's unfortunate that you didn't enjoy it.",
            "Maybe the next one will be better!",
            "Perhaps if the team had more time, they could have come up with a better movie."
            "Different strokes for different folks, I suppose.",
            "Thank you for sharing your honest opinion.",
        ]

        # Prevent consecutive identical responses
        previous_response = st.session_state.get("previous_response", None)
        if class_prediction == 1:
            response_options = [response for response in positive_responses if response != previous_response]
        else:
            response_options = [response for response in negative_responses if response != previous_response]

        if not response_options:
            response_options = positive_responses if class_prediction == 1 else negative_responses

        # Randomly select a response based on sentiment prediction
        selected_response = random.choice(response_options)

        # Store the selected response as the previous response
        st.session_state.previous_response = selected_response

        return selected_response

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main chatbot loop
user_input = st.text_input("You:")
if st.button("Send"):
    if user_input:
        # Add user input to chat history
        st.session_state.chat_history.append(f"You: {user_input}")

        # Generate a response and add it to chat history
        response = generate_response(user_input)
        st.session_state.chat_history.append(f"Chatbot: {response}")

# Display the conversation history
st.text("\n".join(st.session_state.chat_history))

