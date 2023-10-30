
# ............Text Preprocessing..............



import nltk
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download the NLTK stopwords data if you haven't already
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the 20 Newsgroups dataset
newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Initialize the WordNet Lemmatizer and a list of stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Initialize a list to store the preprocessed documents
preprocessed_documents = []

# Preprocess the documents
for document in newsgroups_data.data:
    # Tokenize the text
    words = word_tokenize(document)
    
    # Convert to lowercase and remove stopwords
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    
    # Lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join the words back into a single string
    preprocessed_document = ' '.join(words)
    
    preprocessed_documents.append(preprocessed_document)

# preprocessed_documents now contains the preprocessed text data

# Example: Printing the preprocessed text of the first document
print(preprocessed_documents[0])




# ............Create Term-Document Matrix..............





from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the preprocessed documents to create the TDM
tdm = tfidf_vectorizer.fit_transform(preprocessed_documents)

# Now, 'tdm' is your Term-Document Matrix, where each row represents a document, and each column represents a unique term in the corpus.

# You can access the terms and their corresponding indices using:
terms = tfidf_vectorizer.get_feature_names_out()

# To convert the TDM into a dense matrix and view it as a DataFrame (for visualization purposes):
import pandas as pd
tdm_df = pd.DataFrame(tdm.toarray(), columns=terms)

# Now, 'tdm_df' is a DataFrame representing your Term-Document Matrix.

# Example: Printing the first few rows of the Term-Document Matrix
print(tdm_df.head())






# ............SVD Decomposition..............






from sklearn.decomposition import TruncatedSVD

# Define the number of topics (components) you want to reduce the matrix to
num_topics = 100

# Initialize the TruncatedSVD
svd = TruncatedSVD(n_components=num_topics)

# Fit and transform the TDM using SVD
tdm_reduced = svd.fit_transform(tdm)

# Now, 'tdm_reduced' is your reduced matrix with the specified number of topics.

# You can also access the explained variance by the components to understand the information retained
explained_variance = svd.explained_variance_ratio_.sum()
print(f"Explained Variance: {explained_variance:.2f}")

# 'explained_variance' tells you the proportion of the total variance explained by the selected number of topics.

# Example: Printing the shape of the reduced matrix
print(tdm_reduced.shape)








# ............Topic Exploration..............






# Assuming you have already performed Truncated SVD and obtained 'tdm_reduced' as in the previous response

# Get the singular vectors (components) from Truncated SVD
singular_vectors = svd.components_

# Number of terms and topics
num_terms = len(terms)
num_topics = singular_vectors.shape[0]

# Define the number of top terms to display for each topic
top_terms = 10  # You can adjust this number as needed

# Analyze the terms with the highest weightings in each topic
for topic_idx in range(num_topics):
    # Get the singular vector for the current topic
    singular_vector = singular_vectors[topic_idx]

    # Get the indices of the top terms
    top_term_indices = singular_vector.argsort()[-top_terms:][::-1]

    # Get the actual terms from the indices
    top_topic_terms = [terms[i] for i in top_term_indices]

    # Print the top terms for the current topic
    print(f"Topic {topic_idx + 1}: {', '.join(top_topic_terms)}")










# ............Information Retrieval..............






from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Define your query text
query = "Most of the copies of the text of Acts that we have (including the ones in Vaticanus and Siniaticus) adher pretty closely to the shorter (or Alexandrian) version."

# Preprocess the query (similar to the dataset preprocessing)
query = [query]  # Convert to a list for consistency with the document format
query = tfidf_vectorizer.transform(query)  # Transform the query using the same TF-IDF vectorizer

# Project the query into the LSI space using the same SVD components
query_reduced = svd.transform(query)

# Compute the cosine similarity between the query and the LSI-transformed documents
similarities = cosine_similarity(query_reduced, tdm_reduced)

# Get the indices of the most relevant documents
top_document_indices = similarities[0].argsort()[::-1]

# Define the number of top documents to retrieve
top_documents = 5  # You can adjust this number as needed

# Retrieve and print the most relevant documents
for i in range(top_documents):
    doc_index = top_document_indices[i]
    relevant_document = newsgroups_data.data[doc_index]
    print(f"Relevant Document {i + 1}:\n{relevant_document}\n")









# ............Evaluation..............






from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.datasets import fetch_20newsgroups

# Load the 20 Newsgroups dataset (or your labeled dataset)
newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Preprocess the data (as shown in previous responses)
# You should have 'preprocessed_documents' and 'terms' available

# Create a Term-Document Matrix (TDM)
tfidf_vectorizer = TfidfVectorizer()
tdm = tfidf_vectorizer.fit_transform(preprocessed_documents)

# Perform Truncated SVD to reduce dimensionality
num_topics = 100
svd = TruncatedSVD(n_components=num_topics)
tdm_reduced = svd.fit_transform(tdm)

# Apply clustering, e.g., K-Means, to the reduced data
num_clusters = 20  # You can adjust the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
clusters = kmeans.fit_predict(tdm_reduced)

# Evaluate clustering performance using metrics
# For NMI and silhouette score, you need the ground truth labels
true_labels = newsgroups_data.target  # Replace with your actual labels

nmi_score = normalized_mutual_info_score(true_labels, clusters)
silhouette_avg = silhouette_score(tdm_reduced, clusters)

print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Purity is a measure of how well the clusters match the ground truth
from collections import Counter
from scipy.stats import mode

def purity_score(y_true, y_pred):
    contingency_matrix = -np.ones((max(y_pred) + 1, max(y_true) + 1), dtype=np.int64)
    for i in range(len(y_pred)):
        contingency_matrix[y_pred[i], y_true[i]] += 1
    return np.sum(np.max(contingency_matrix, axis=0)) / len(y_pred)

purity = purity_score(true_labels, clusters)
print(f"Purity: {purity:.4f}")






