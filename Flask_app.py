from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from transformers import AutoTokenizer, AutoModel
from pymilvus import (Collection,connections)
from flask import Flask, request, jsonify
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")
import torch.nn.functional as F
import pandas as pd
import warnings
import torch


#connect to vector database
ENDPOINT="https://in03-0008120f0b8b227.serverless.gcp-us-west1.cloud.zilliz.com"
connections.connect(
   uri=ENDPOINT,
   token="2e021870ed0adebedcf7a869bc5df9905510a5d53114ee7aff6fd8f08d7799bb511c6de82f13d3c4bf45740ca0f0d4d26c0fdcb7")
collection_name = "jp3"
collection = Collection(name=collection_name)

def lower_transform(text):
    return text.lower()

def remove_excess_whitespace(text):
    stripped_text = text.strip()
    cleaned_text = ' '.join(stripped_text.split())
    return cleaned_text



def search (embeddings,collection):
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}    
    results = collection.search(
    
    data=[embeddings], 
    anns_field="vector",  
    output_fields=["email","description"],
    limit=3,
    param=search_params
)
    for hits in results:
        
        emails=[]
        descriptions=[]
        for hit in hits:
            # gets the value of an output field specified in the search request.
            # dynamic fields are supported, but vector fields are not supported yet.    
            emails.append(hit.entity.get('email'))
            descriptions.append(hit.entity.get('description'))

    return emails,descriptions

# Initialize the tokenizer and BERT model


# Load the 'all-roberta-large-v1' model

class BertEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='sentence-transformers/all-roberta-large-v1', device=None):
        # Initialize the device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load the tokenizer and model from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,clean_up_tokenization_spaces=False)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Tokenize the input text
        X=[X]
        encoded_input = self.tokenizer(X, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        # Generate token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Perform mean pooling to get sentence embeddings
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize the embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings.cpu().numpy().flatten()

# Initialize the transformers
lowercase_transformer = FunctionTransformer(lower_transform, validate=False)
whitespace_transformer = FunctionTransformer(remove_excess_whitespace, validate=False)
bert_embedding_transformer = BertEmbeddingTransformer()

# Create the pipeline
embeddings_pipeline = Pipeline([
    ('lowercase', lowercase_transformer),
    ('whitespace', whitespace_transformer),
    ('bert_embedding', bert_embedding_transformer)
])


#defining endpoints
app = Flask(__name__)

@app.route('/store', methods=['POST'])
def get_description():
    text = request.get_json()
    data = pd.DataFrame(text)
    
    if "email" not in data.columns or "description" not in data.columns:
        return jsonify({"error": "Input JSON must contain 'email' and 'description' fields"}), 400
    
    email = data["email"].values
    descriptions = data["description"].values
    embeddings = [embeddings_pipeline.transform(desc) for desc in descriptions]
    try:
        data_rows = []
        data_rows.extend([
        {"vector": embeddings[0],
            "email":email[0],
            "description":descriptions[0]}
        ])
        collection.insert(data_rows)
        collection.flush()


        return jsonify({"status":"stored"})
    except Exception as x:
        return jsonify({"error": str(x)}),501
    




@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    try:
        text = request.get_json()
        data = pd.DataFrame(text)
        if "description" not in data.columns:
          return jsonify({"error": "Input JSON must contain 'description' field"}), 400
        description = data["description"].values[0] 
        embeddings = embeddings_pipeline.transform(description)  
        emails=[]
        descriptions=[]
        emails,descriptions=search(embeddings,collection)
        return jsonify({"emails":emails,"descriptions":descriptions })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/update', methods=['POST'])
def update_data():
    text = request.get_json()
    data = pd.DataFrame(text)
    
    if "email" not in data.columns or "description" not in data.columns:
        return jsonify({"error": "Input JSON must contain 'email' and 'description' fields"}), 400
    
    email = data["email"].values[0]
    descriptions = data["description"].values[0]
    embeddings = embeddings_pipeline.transform(descriptions) 
    try:
        data_rows = []
        data_rows.extend([
        {"vector": embeddings,
            "email":email,
            "description":descriptions}
        ])
        collection.upsert(data_rows)
        collection.flush()


        return jsonify({"status":"updated"})
    except Exception as x:
        return jsonify({"error": str(x)}),500    


@app.route('/delete', methods=['POST'])
def delete_data():
    try:
        text = request.get_json()
        data = pd.DataFrame(text)
        
        if "email" not in data.columns :
            return jsonify({"error": "Input JSON must contain 'email'  field"}), 400
        
        email = data["email"].values[0]
        expre=f'email == "{email}"' 
        results = collection.query(expre, output_fields=["email"], limit=1)
        if len(results)==0:
            return jsonify({"error": "No data found for the given email."}), 500
        
        res = collection.delete(
        
            expr=expre 
                    
        )
        return jsonify({"status":"deleted"})
 
    except Exception as e:
        return jsonify({"error": str(e)}), 501

    
if __name__ == "__main__":
    app.run(debug=True)
