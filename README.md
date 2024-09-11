### Overview
--------

This platform allows users to store or update their job application data, receive job applicant recommendations based on job descriptions, and delete their stored data. The platform leverages a BERT-based text processing pipeline and Milvus for efficient storage and retrieval of embeddings.

### Python Package Requirements:

1.  **Flask**: To create the API endpoints.

2.  **pandas**: For handling and manipulating data.

3.  **torch**: To run the BERT model from HuggingFace (PyTorch is required).

4.  **transformers**: For loading and using the pre-trained BERT model.

5.  **pymilvus**: To connect and interact with the Milvus vector database.

6.  **scikit-learn**: For using the pipeline, base estimators, and transformers.

7.  **streamlit**: For building an interactive web application

8.  **warnings**: For suppressing warnings (comes with Python by default, so no need to install).

### Key Components:

### 1\. Database Connection

For the project, I used **Zilliz Cloud** to host the Milvus vector database. By utilizing this cloud-based solution, the vector database remains accessible to the website during deployment, ensuring seamless integration and real-time access to data. Zilliz Cloud was chosen due to its ability to handle large-scale data, making it ideal for a scalable job-matching platform. Milvus is specifically designed to store and search through millions of vector embeddings efficiently, allowing the platform to accommodate a growing number of users and recommendations without sacrificing performance.

By hosting Milvus on Zilliz Cloud, the project benefits from:

-   **Scalability**: As the user base grows, Milvus can manage millions of vector embeddings while maintaining quick response times during searches and queries.
-   **Cloud Accessibility**: Hosting on Zilliz Cloud ensures that the vector database is accessible from any location, enabling efficient real-time interaction from the deployed website.
-   **High Availability and Maintenance**: Leveraging a managed cloud service like Zilliz ensures that the database is highly available, with automatic updates and maintenance that reduce the operational burden.

### 2\. **Text Cleaning Functions**

The text cleaning functions perform basic preprocessing to prepare the text for embedding.

-   **`lower_transform(text)`**: Converts all text to lowercase to maintain consistency during tokenization.


-   **`remove_excess_whitespace(text)`**: Strips leading/trailing whitespace and removes extra spaces between words.



### 3\. **BERT Embedding Transformer**

A custom transformer is created by subclassing `BaseEstimator` and `TransformerMixin` to integrate BERT embeddings into the `scikit-learn` pipeline.

-   **`BertEmbeddingTransformer`**: This class is responsible for:
    -   Loading the tokenizer and model using Hugging Face's `transformers` library.
    -   Generating token embeddings for input text.
    -   Applying **mean pooling** to convert token embeddings into sentence embeddings.
    -   Normalizing the sentence embeddings using L2 normalization.



Key Points:

-   **Mean Pooling**: Converts token-level embeddings into a single vector representing the entire sentence.
-   **Normalization**: Ensures embeddings are unit vectors (L2 normalization).

### 4\. **Pipeline Creation**

A `scikit-learn` pipeline is built using the following components:

-   **Lowercase Transformer**: Converts the input text to lowercase.
-   **Whitespace Transformer**: Removes excess whitespace from the text.
-   **BERT Embedding Transformer**: Generates sentence embeddings from the cleaned text.



### Key Steps in the Pipeline:

1.  **Lowercase Transformation**: Converts all text to lowercase for uniform tokenization.
2.  **Whitespace Removal**: Strips unnecessary spaces and normalizes spacing within the text.
3.  **BERT Embedding Generation**: Generates meaningful embeddings by:
    -   Tokenizing the input text.
    -   Generating token embeddings using a pre-trained BERT model.
    -   Applying mean pooling and normalizing the embeddings.

* * * * *

### Benefits:

-   **Efficient**: All transformations, from text preprocessing to embedding generation, are handled within a single pipeline.


# API Documentation

This API allows users to store, update, retrieve recommendations, and delete job-related data using email and description fields. All data should be passed via JSON in the request body.


## Flask API Endpoints

### 1. Store Date (`/store`)


- **URL**: `/store`
- **Method**: `POST`
- **Description**:`Store a user's email and description into the database, along with its embedding.`
#### **Request Body**:
  ```json
  {
    "email": "user@example.com",
    "description": "Experienced data scientist skilled in machine learning."
  }
```
#### **Success Response**:
- **Code**: `200 OK`
- **Content**:
  ```json
  {
  "status":"stored"
  }
  ```
#### **Error Responses**:
- **Code**: `400 Bad Request`
- **Content**:
  ```json
  {
   "error": "Input JSON must contain 'email' and 'description' fields"
  }
  ```
- **Code**: `501 Internal Server Error`
- **Content**:
  ```json
  {
  "error": "error message" 
  }
  ```
### 2. Get Recommendations (`/get_recommendation`)


- **URL**: `/get_recommendation`
- **Method**: `POST`
- **Description**:`Retrieve a list of recommended applicants based on the provided job description.`
#### **Request Body**:
  ```json
  {
    "description": "Job description"
  }
```
#### **Success Response**:
- **Code**: `200 OK`
- **Content**:
  ```json
   {
    "email": "user@example.com",
    "description": "Description"
  }
  ```
#### **Error Responses**:
- **Code**: `400 Bad Request`
- **Content**:
  ```json
  {
  "error": "Input JSON must contain 'description' field"
  }
  ```
- **Code**: `500 Internal Server Error`
- **Content**:
  ```json
  {
  "error": "error message" 
  }
  ```
### 3. Update Data(`/update`)


- **URL**: `/get_recommendation`
- **Method**: `POST`
- **Description**:`Update a user's description and embeddings in the database.`
#### **Request Body**:
  ```json
  {
    "email": "user@example.com",
    "description": "Updated description"
  }
```
#### **Success Response**:
- **Code**: `200 OK`
- **Content**:
  ```json
   {
     "status": "updated"

  }
  ```
#### **Error Responses**:
- **Code**: `400 Bad Request`
- **Content**:
  ```json
  {
  "error": "Input JSON must contain 'email' and 'description' fields"
  }
  ```
- **Code**: `500 Internal Server Error`
- **Content**:
  ```json
  {
  "error": "error message" 
  }
  ```
### 4. Delete Data (`/delete`)


- **URL**: `/get_recommendation`
- **Method**: `POST`
- **Description**:`Delete a user's data based on their email.`
#### **Request Body**:
  ```json
  {
    "email": "user@example.com",
  }
```
#### **Success Response**:
- **Code**: `200 OK`
- **Content**:
  ```json
   {
     "status": "updated"

  }
  ```
#### **Error Responses**:
- **Code**: `400 Bad Request`
- **Content**:
  ```json
  {
  "error": "Input JSON must contain 'email'  field"
  }
  ```
- **Code**: `500 Internal Server Error`
- **Content**:
  ```json
  {
  "error": "No data found for the given email."
  }
  ```
- **Code**: `501 Internal Server Error`
- **Content**:
  ```json
  {
  "error": "error message"
  }
  ```


  ## Error Handling


- **400 Bad Request**: `Indicates that the input data is missing required fields.`
- **500/501 Server Error**: `Indicates server-side errors, such as failing to retrieve data or perform actions like insertion, deletion, etc`
  
## Notes
-----

1.  **All data** must be sent in the request body in **JSON format**.
2.  **Email** is the primary identifier used to manage stored data.
3.  **Embedding-based search** is used for recommendations and data storage.
4.  Ensure the server URL is correctly specified when calling the API.
5.  I used post request method only to ensure that no personal data will be sent via URL


## Streamlit App
--------

### 1\. **Store/Update Data**

-   **Description**: Users can enter their job-related skills or description along with their email. The system checks if the email exists in the database (Milvus), and either updates or inserts the data accordingly.
-   **Function used**:'get_description' function 
-   **Input Fields**:
    -   **Description**: The user's job-related description (skills and experience).
    -   **Email**: The user's email address.
-   **Function flow**: The get_description function processes the input description, checks whether data is stored along with the email address, generates embeddings, and stores them in Milvus. If the email already exists, the system updates the existing data.
-   **UI Element**:
    -   The input form is in an expandable section with two fields one for email and the other for description.
    -   A button triggers the "Store My Data" action which calls get_description function.
    -  **Output**: it returns whether the data has been updated or stored 

### 2\. **Get Applicant Recommendations**

-   **Description**: Users enter a job description to find applicants that match the description.
-   **Function used**:get_recommendation function
-   **Input Fields**:
    -   **Job Description**: A description of the job for which you are searching for matching applicants.
-   **Function**: The function generates the embeddings to compare the job description with stored applicant data and returns the top three recommendations based on similarity.
-   **UI Element**:
    -   The input form for the job description is in an expandable section with one field to input job description .
    -   A button triggers the "Get Recommendations" action.
-   **Output**: The matching applicants' emails and descriptions are displayed.

### 3\. **Delete My Data**

-   **Description**: Users can delete their stored data from the platform using their email.
-   **Function used**: delete_data
-   **Input Fields**:
    -   **Email**: The email corresponding to the data you wish to delete.
-   **Function**: The used function searches for the email in the Milvus collection and deletes the corresponding record if it exists.
-   **UI Element**:
    -   The input form is in an expandable section with one field to enter the email address.
    -   A button triggers the "Delete My Data" action.
-   **Output**:it returns whether the data has been deleted or no data found for the given email


Note
--------
all features check whether all required input field has data if do so it calls the function and if not it shows a message to the user to enter all fields

