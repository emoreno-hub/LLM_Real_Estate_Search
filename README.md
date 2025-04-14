# RealtorGPT
### Smarter home search through AI-driven personalization
I built this project to explore how AI can make the property search experience more personalized and meaningful for buyers. By combining large language models (LLMs) with vector databases, the app transforms standard real estate listings into tailored narratives that align with each buyer’s unique preferences and lifestyle. The goal is to move beyond basic filters and create a smarter, more engaging home search powered by semantic understanding.

## Methodology
To achieve this, the project utilizes LangChain and ChromaDB:
- **LangChain:** LangChain provides a structured framework to interact with Large Language Models (LLMs). It simplifies prompt engineering, response parsing, and integration with external data sources, making it easier to generate high-quality real estate listings dynamically.

- **ChromaDB:** ChromaDB serves as a vector database to store and retrieve property listings based on semantic similarity. By converting property descriptions into vector embeddings, the system can quickly find and recommend relevant listings based on user queries, improving search accuracy beyond simple keyword matching.

## Core Components of the RealtorGPT Application

### Understanding Buyer Preferences

- Buyers will input their requirements and preferences, such as location, property type, budget, amenities, and lifestyle choices.
- The application uses **LLMs (powered by LangChain)** to interpret these inputs in natural language, understanding nuanced requests beyond basic filters.

### Integrating with a Vector Database

- Realtor GPT connects with **ChromaDB**, a vector databases where all available property listings are stored.
- The system generates and stores vector embeddings for each listing, enabling semantic search that considers deeper aspects like neighborhood vibes, architectural styles, and proximity to specific amenities..

### Streamlit App
The user interface is built with Streamlit to make the experience interactive. Buyers enter their preferences and the app displays listings with AI-enhanced descriptions that are written in real-time to emphasize features most relevant to the buyer.
<img src="app_screenshot.PNG" alt="Streamlit app screenshot" width="50%">
![StreamlitApp](app_screenshot2.PNG)

### Personalized Listing Description Generation

- Each generated real estate listing follows a structured listing template, ensuring consistency in details such as price, bedrooms, and neighborhood descriptions. The listing template is as follows:
```bash
Neighborhood: [Neighborhood Name]
Price: [$Amount]
Bedrooms: [Number]
Bathrooms: [Number]
House Size: [Square Footage]

Description: [A unique and engaging property description, highlighting features like design, amenities, and lifestyle benefits.]

Neighborhood Description: [A brief summary of the surrounding neighborhood, including community highlights, nearby attractions, and lifestyle perks.]
```

- Once a buyer's preferences are identified, the LLM dynamically rewrites property descriptions to emphasize features most relevant to them (e.g., highlighting proximity to schools for families or showcasing smart home technology for tech-savvy buyers).
- The personalized descriptions maintain factual accuracy while enhancing engagement by tailoring the narrative to the buyer’s interests.

### Listing Presentation

- Output the personalized listing(s) as a text description of the listing.

## How to execute
This project supports both Docker-based and manual setup. The recommended method is using Docker for a consistent and portable environment.

### Option 1: Run with Docker (Recommended)
1 - Clone this repository
```bash
https://github.com/emoreno-hub/Realtor_GPT.git
```

2 - Create a .env file
```bash
OPENAI_API_KEY = 'YOUR API KEY'
```

3 - Build and run the Streamlit app
```bash
docker-compose up --build
```

### Option 2: Manual Python Setup
1 - Clone a virtual environment

```bash
conda create --name agent --python==3.9.18
```

2 - Install the requirements

```bash
pip install -r app/requirements.txt
```

3 - Create a .env file
```bash
OPENAI_API_KEY = 'YOUR API KEY'
```

## Author
**Eric Moreno**

