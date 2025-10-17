import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import threading

# --- Data Loading and Preprocessing ---

def load_and_process_data():
    """Loads and processes the movie data in a background thread."""
    global data, cosine_sim, indices
    
    status_label.config(text="Status: Loading dataset...")
    try:
        movies = pd.read_csv('movies.csv')
    except FileNotFoundError:
        status_label.config(text="Error: 'movies.csv' not found. Please place it in the same folder.")
        return

    # Select, clean, and prepare data
    required_columns = ['title', 'genres', 'keywords', 'cast', 'directors', 'writers', 'overview']
    movies = movies[required_columns]
    movies.fillna('', inplace=True)
    movies.drop_duplicates(inplace=True)
    movies['tag'] = movies.apply(lambda row: ' '.join(row.astype(str)), axis=1)
    
    # Using 15,000 movies for a good balance of speed and variety
    data = movies[['title', 'tag']].head(15000).copy()

    status_label.config(text="Status: Preprocessing text data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
        tokens = word_tokenize(text)
        return " ".join([word for word in tokens if word not in stop_words])

    data['processed_tag'] = data['tag'].apply(preprocess_text)

    # TF-IDF Vectorization and Similarity Calculation
    status_label.config(text="Status: Calculating movie similarities...")
    tfidf = TfidfVectorizer(max_features=15000)
    tfidf_matrix = tfidf.fit_transform(data['processed_tag'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(data.index, index=data['title']).drop_duplicates()
    
    # Enable UI components now that data is ready
    movie_entry.config(state='normal')
    recommend_button.config(state='normal')
    status_label.config(text="Status: Ready! Enter a movie name.")
    results_text.delete('1.0', tk.END)
    results_text.insert(tk.END, "Enter a movie title above and click 'Get Recommendations'.")


# --- Recommendation Function ---

def get_recommendations_for_title(exact_title):
    """Generates recommendations for a given exact movie title."""
    idx = indices[exact_title]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return data['title'].iloc[movie_indices]

def handle_search():
    """Handles the user's search query, finds the best match, and displays results."""
    query = movie_entry.get().strip()
    if not query:
        results_text.delete('1.0', tk.END)
        results_text.insert(tk.END, "Please enter a movie name to search.")
        return

    # Find the best match for the query (case-insensitive search)
    # This finds all movies where the title contains the user's query
    matches = data[data['title'].str.lower().str.contains(query.lower())]

    if matches.empty:
        results_text.delete('1.0', tk.END)
        results_text.insert(tk.END, f"Sorry, no movie found matching '{query}'.\nPlease check the spelling or try another movie.")
        return
    
    # Pick the first match as the movie to recommend from
    best_match_title = matches['title'].iloc[0]
    
    # Generate recommendations for the found movie
    recommended_movies = get_recommendations_for_title(best_match_title)

    # Format and display the output
    results_text.delete('1.0', tk.END)
    output = f"Showing recommendations for '{best_match_title}':\n\n"
    output += "\n".join([f"{i+1}. {title}" for i, title in enumerate(recommended_movies)])
    results_text.insert(tk.END, output)


# --- UI Setup ---

# Main window
root = tk.Tk()
root.title("🎬 The Film Oracle")
root.geometry("500x450")
root.resizable(False, False)

# --- Theme Configuration ---
BG_COLOR = "#2E2E2E"
FG_COLOR = "#F0F0F0"
ENTRY_BG = "#3C3C3C"
BUTTON_BG = "#555555"
BUTTON_ACTIVE = "#6A6A6A"

root.configure(bg=BG_COLOR)

# Configure ttk styles
style = ttk.Style(root)
style.theme_use('clam') # 'clam' is a good base theme for customization

style.configure('.', 
    background=BG_COLOR, 
    foreground=FG_COLOR,
    fieldbackground=ENTRY_BG,
    borderwidth=0,
    lightcolor=BUTTON_BG,
    darkcolor=BUTTON_BG
)

style.configure('TFrame', background=BG_COLOR)
style.configure('TLabel', background=BG_COLOR, foreground=FG_COLOR, font=('Helvetica', 10))
style.configure('Header.TLabel', font=("Helvetica", 16, "bold")) # Custom style for the title

style.configure('TButton', 
    background=BUTTON_BG, 
    foreground=FG_COLOR, 
    font=('Helvetica', 10, 'bold'),
    borderwidth=0,
    focusthickness=0,
    focuscolor='none'
)
style.map('TButton', background=[('active', BUTTON_ACTIVE)])

style.configure('TEntry', 
    fieldbackground=ENTRY_BG,
    foreground=FG_COLOR,
    insertcolor=FG_COLOR, # Cursor color
    borderwidth=1
)

# --- End of Theme Configuration ---

# Main frame
frame = ttk.Frame(root, padding="10")
frame.pack(fill=tk.BOTH, expand=True)

# Title label
title_label = ttk.Label(frame, text="The Film Oracle", style='Header.TLabel')
title_label.pack(pady=5)

# Entry for movie search
instruction_label = ttk.Label(frame, text="Enter a Movie Name:")
instruction_label.pack(pady=(10, 2))

movie_entry = ttk.Entry(frame, width=50, state='disabled')
movie_entry.pack()

# Recommend button
recommend_button = ttk.Button(frame, text="Get Recommendations", command=handle_search, state='disabled')
recommend_button.pack(pady=10)

# Results text box
results_text = tk.Text(
    frame, 
    height=12, 
    width=60, 
    wrap=tk.WORD, 
    relief=tk.SOLID, 
    borderwidth=1,
    bg=ENTRY_BG,
    fg=FG_COLOR,
    highlightthickness=1,
    highlightbackground=BUTTON_BG,
    insertbackground=FG_COLOR # Cursor color for Text widget
)
results_text.pack(pady=5)
results_text.insert(tk.END, "Please wait, the system is loading...")

# Status bar
status_label = ttk.Label(frame, text="Status: Initializing...", relief=tk.SUNKEN, anchor=tk.W)
status_label.pack(side=tk.BOTTOM, fill=tk.X)


# --- Main Execution ---

if __name__ == "__main__":
    # Run data processing in a separate thread to keep the UI from freezing
    processing_thread = threading.Thread(target=load_and_process_data)
    processing_thread.start()
    
    # Start the Tkinter event loop
    root.mainloop()