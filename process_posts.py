import pandas as pd
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.util import ngrams # NLTK is used to generate and filter n-grams (word pairs) from the text.
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

stop_words = set(stopwords.words('english'))
# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to calculate VADER sentiment score
def get_vader_sentiment(text):
    return sia.polarity_scores(text)['compound']

# Function to generate and filter n-grams
def generate_filtered_ngrams(text, n=2):
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    generated_ngrams = ngrams(filtered_words, n)
    return ', '.join([' '.join(gram) for gram in generated_ngrams])

def co_occurence_table(data,filename):
    # Create a new DataFrame for the co-occurrence table
    co_occurrence = pd.DataFrame()

    # Fill the new DataFrame with the required data
    co_occurrence['user'] = data['user']
    co_occurrence['author'] = data['author']
    co_occurrence['comment'] = data['comment']
    co_occurrence['title'] = data['title']
    co_occurrence['post_text'] = data['post_text']
    co_occurrence['comm_date'] = data['comm_date']
    co_occurrence['subreddit'] = data['url'].apply(lambda url: url.split('/')[4] if len(url.split('/')) > 4 else 'Unknown')

    # Adding N-grams and sentiment analysis results
    co_occurrence['ngrams.title'] = co_occurrence['title'].astype(str).apply(lambda x: generate_filtered_ngrams(x, 2))
    co_occurrence['ngrams.comment'] = co_occurrence['comment'].astype(str).apply(lambda x: generate_filtered_ngrams(x, 2))
    co_occurrence['vader.title'] = co_occurrence['title'].astype(str).apply(get_vader_sentiment)
    co_occurrence['vader.comment'] = co_occurrence['comment'].astype(str).apply(get_vader_sentiment)

    # Handle missing values if necessary (e.g., with empty strings)
    co_occurrence.fillna('', inplace=True)

    # Save the co-occurrence table with additional data to a new CSV file
    output_file_path = filename+'.csv'  # Replace with your desired output file path
    co_occurrence.to_csv(output_file_path, index=False)

    print("Enhanced co-occurrence table with N-grams created and saved to:", output_file_path)
    return 


def plot_entries_and_threads(dataframe):
    """
    Plots the total volume of entries and the number of new threads over time.

    Parameters:
    - dataframe: A pandas DataFrame with columns 'post_date' and 'user'.

    The function creates two plots: 
    - Total number of entries (posts and comments) per day.
    - Total number of new threads per day (assuming a new thread for each entry).
    """

    # Convert 'post_date' to datetime format
    dataframe['post_date'] = pd.to_datetime(dataframe['post_date'])

    # Resample to get daily counts
    daily_counts = dataframe.resample('D', on='post_date').id.count()

    # Assuming each entry is a new thread
    dataframe['is_new_thread'] = dataframe['user'] == dataframe['user']
    daily_new_thread_counts = dataframe[dataframe['is_new_thread']].resample('D', on='post_date').id.count()

    # Creating the plots
    plt.figure(figsize=(12, 12))

    # Plot for Total Entries
    plt.subplot(2, 1, 1)
    sns.lineplot(data=daily_counts, color='blue')
    plt.title('Total Volume of Entries Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Entries')
    plt.grid(True)

    # Plot for New Threads
    plt.subplot(2, 1, 2)
    sns.lineplot(data=daily_new_thread_counts, color='green')
    plt.title('Number of New Threads Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of New Threads')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_entries_and_threads_by_subreddit(dataframe):
    """
    Generates a grid of plots for each subreddit in the dataframe, showing the total volume of entries and 
    the number of new threads over time.

    Parameters:
    - dataframe: A pandas DataFrame with columns 'post_date', 'user', and 'subreddit'.
    """
    
    # Convert 'post_date' to datetime format
    dataframe['post_date'] = pd.to_datetime(dataframe['post_date'])

    # Get unique subreddits
    unique_subreddits = dataframe['subreddit'].unique()
    num_subreddits = len(unique_subreddits)

    plt.figure(figsize=(18, 6 * num_subreddits))  # Adjusted for better readability

    for i, subreddit in enumerate(unique_subreddits, 1):
        # Filter data for the current subreddit
        subreddit_data = dataframe[dataframe['subreddit'] == subreddit]

        # Resample to get daily counts for total entries
        daily_counts_subreddit = subreddit_data.resample('D', on='post_date').id.count()

        # Assuming each entry is a new thread
        subreddit_data['is_new_thread'] = subreddit_data['user'] == subreddit_data['user']
        daily_new_thread_counts_subreddit = subreddit_data[subreddit_data['is_new_thread']].resample('D', on='post_date').id.count()

        # Plot for Total Entries in this subreddit
        plt.subplot(num_subreddits, 2, 2 * i - 1)
        sns.lineplot(data=daily_counts_subreddit, color='blue')
        plt.title(f'Total Entries Over Time in {subreddit}')
        plt.xlabel('Date')
        plt.ylabel('Number of Entries')
        plt.grid(True)

        # Plot for New Threads in this subreddit
        plt.subplot(num_subreddits, 2, 2 * i)
        sns.lineplot(data=daily_new_thread_counts_subreddit, color='green')
        plt.title(f'New Threads Over Time in {subreddit}')
        plt.xlabel('Date')
        plt.ylabel('Number of New Threads')
        plt.grid(True)

    plt.tight_layout()
    plt.show()