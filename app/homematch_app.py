import os
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, NonNegativeInt
from typing import List
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from fastapi.encoders import jsonable_encoder
from dotenv import load_dotenv

import streamlit as st

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# initialize OpenAI LLM
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.0,
    openai_api_key=openai_api_key
)
# initialize ChromaDB path
CHROMA_PATH = "chroma"
# initialize embeddings and text splitter
embeddings = OpenAIEmbeddings() # Initialize the embeddings
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=100)  # Adjust chunk_size and chunk_overlap as needed

# setup instructions and template for listings
default_instruction = "Generate at least 10 real estate listings in the same format as shown below. Repeat the listing structure exactly for each entry."


default_template = \
"""
Neighborhood: Green Oaks
Price: $800,000
Bedrooms: 3
Bathrooms: 2
House Size: 2,000 sqft

Description: Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom, 2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces, highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family. Embrace sustainable living without compromising on style in this Green Oaks gem.

Neighborhood Description: Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze.
"""

# define the structure of the data for the real estate listings
class RealEstateListing(BaseModel):
    """
    A real estate listing.
    
    Attributes:
    - neighborhood: str
    - price: NonNegativeInt
    - bedrooms: NonNegativeInt
    - bathrooms: NonNegativeInt
    - house_size: NonNegativeInt
    - description: str
    - neighborhood_description: str
    """
    
    neighborhood: str = Field(description="The neighborhood where the property is located")
    price: NonNegativeInt = Field(description="The price of the property in USD")
    bedrooms: NonNegativeInt = Field(description="The number of bedrooms in the property")
    bathrooms: NonNegativeInt = Field(description="The number of bathrooms in the property")
    house_size: NonNegativeInt = Field(description="The size of the house in square feet")
    description: str = Field(description="A description of the property")
    neighborhood_description: str = Field(description="A description of the neighborhood.")  

class ListingCollection(BaseModel):
    """
    A collection of real estate listings.
    
    Attributes:
    - listings: List[RealEstateListing]
    """
    
    listings: List[RealEstateListing] = Field(description = "A list of real estate listings")

parser = PydanticOutputParser(pydantic_object = ListingCollection)

# setup the prompt template
prompt = PromptTemplate(
    input_variables=["instruction", "template"],
    partial_variables={"format_instructions": parser.get_format_instructions},
    template=(
        "{instruction}\n\n"
        "{template}\n\n"
        "{format_instructions}\n\n"
        "Now generate 10 unique listings following the format above."
    )
)

# Function to simulate initial listing generation
def generate_listings(instruction, listing_template):
    query = prompt.format(instruction=instruction,template=listing_template)
    response = llm.predict(query)
    result = parser.parse(response)
    df = pd.DataFrame(jsonable_encoder(result.listings))
    return df

# create vector database from listings
def create_vector_store(df):
    documents = []
    for index, row in df.iterrows():
        chunks = splitter.split_text(row['description'])
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata={'id': str(index)}))

    db = Chroma.from_documents(documents, embeddings, persist_directory=CHROMA_PATH)
    db.persist()
    return db

# Preference-based search and augmentation
def construct_query(preferences):
    return (
        f"Find properties with {prefs['bedrooms']} bedrooms, {prefs['bathrooms']} bathrooms, "
        f"located in {prefs['location']} within the price range of {prefs['price_range']} "
        f"and at least {prefs['house_size']} sqft."
    )

def query_vector_database(preferences, db, embeddings, k=20):
    """
    Query the vector database using buyer preferences.

    :param preferences: Dictionary containing buyer preferences.
    :param db: Vector database instance.
    :param embeddings: Embedding model to convert the query into a vector.
    :param k: Number of results to retrieve.
    :return: List of retrieved documents.
    """

    query_text = construct_query(preferences)
    query_vector = embeddings.embed_query(query_text)
    try:
        results = db.similarity_search_by_vector(query_vector, k=k)
        if not results:
            print("No matching properties found.")
        return results
    except Exception as e:
        print(f"An error occurred while querying the database: {e}")
        return []

def augment_listing(description, preferences):
    prompt_text = (
        f"Buyer Preferences:\n"
        f" - Bedrooms: {preferences['bedrooms']}\n"
        f" - Bathrooms: {preferences['bathrooms']}\n"
        f" - Location: {preferences['location']}\n"
        f" - Price Range: {preferences['price_range']}\n"
        f" - Minimum House Size: {preferences['house_size']} sqft\n\n"
        f"Property Description:\n"
        f"{description}\n\n"
        f"Task: Rewrite the property description to subtly emphasize features that align with the buyer's preferences. "
        f"Ensure factual accuracy while making the description more appealing to the buyer."
    )
    try:
        response = llm([HumanMessage(content=prompt_text)])
        return response.content.strip()
    except Exception as e:
        return f"(Augmentation failed: {e})"


# ───────────────────────────────────────────────
# STREAMLIT INTERFACE
# ───────────────────────────────────────────────

st.set_page_config(page_title="HomeMatch", page_icon="🏡")
st.title("🏡 HomeMatch: AI-Augmented Property Listings")

st.markdown("""
#### Welcome to **HomeMatch**
This AI-powered tool demonstrates how natural language processing and vector search can be used to personalize the home buying experience.

Simply enter your desired number of bedrooms, bathrooms, location, price range, and minimum square footage.  
The app will generate sample listings using a large language model, store them in a vector database, and surface the most relevant matches based on your preferences.

Each listing is then rewritten to emphasize the features you're looking for — offering a glimpse at how AI can enhance property discovery in a more personalized, engaging way.
""")

st.header("🏘️ Buyer Preferences")
with st.form("prefs_form"):
    bedrooms = st.text_input("Bedrooms")
    bathrooms = st.text_input("Bathrooms")
    location = st.text_input("Preferred Location")
    price_range = st.text_input("Price Range (e.g., 500000-800000)")
    house_size = st.text_input("Minimum House Size (sqft)")
    submitted = st.form_submit_button("Generate and Personalize Listings")

if submitted:
    prefs = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "location": location,
        "price_range": price_range,
        "house_size": house_size
    }

    with st.spinner("Generating listings, building vector store, and retrieving matches..."):

        try:
            # Step 1: Generate Listings
            df_listings = generate_listings(default_instruction, default_template)
            st.session_state.df_listings = df_listings

            # Step 2: Build Vector DB
            db = create_vector_store(df_listings)
            st.session_state.db = db

            # Step 3: Query Vector DB
            results = query_vector_database(prefs, db, embeddings)

            # Step 4: Deduplicate and Show Results
            if not results:
                st.warning("No matching listings found.")
            else:
                seen_ids = set()
                augmented_results = []

                for doc in results:
                    doc_id = doc.metadata.get("id", "N/A")
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)

                        original_description = doc.page_content
                        augmented = augment_listing(original_description, prefs)

                        augmented_results.append({
                            "id": doc_id,
                            "original_description": original_description,
                            "augmented_description": augmented
                        })

                if not augmented_results:
                    st.warning("All matching results were duplicates.")
                else:
                    st.success(f"Found {len(augmented_results)} unique listings.")
                    for result in augmented_results:
                        st.markdown("----")
                        st.markdown(f"**Listing ID**: {result['id']}")
                        # st.markdown(f"**Original Description**: {result['original_description']}")
                        st.markdown("**📝 Augmented Description:**")
                        st.write(result['augmented_description'])
        except Exception as e:
            st.error(f"An error occurred: {e}")