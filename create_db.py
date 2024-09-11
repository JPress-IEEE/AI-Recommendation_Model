from pymilvus import (
   FieldSchema, DataType, 
   CollectionSchema, Collection,connections)

ENDPOINT="https://in03-0008120f0b8b227.serverless.gcp-us-west1.cloud.zilliz.com"
connections.connect(
   uri=ENDPOINT,
   token="2e021870ed0adebedcf7a869bc5df9905510a5d53114ee7aff6fd8f08d7799bb511c6de82f13d3c4bf45740ca0f0d4d26c0fdcb7")



## 1. Define a minimum expandable schema.
fields = [
   FieldSchema("vector", DataType.FLOAT_VECTOR, dim=1024),
   FieldSchema("email", DataType.VARCHAR,max_length=100, is_primary=True),
   FieldSchema("description", DataType.VARCHAR,max_length=10000),
]

schema = CollectionSchema(
   fields,
   enable_dynamic_field=True,
)

## 2. Create a collection.
mc = Collection("jp3", schema)

## 3. Index the collection.
mc.create_index(
   field_name="vector",
   index_params={
       "Index_type": "AUTOINDEX",
       "Metric_type": "L2",
       }
)

mc.load()