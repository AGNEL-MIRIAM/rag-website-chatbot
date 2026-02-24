import os
from flask import Flask, render_template, request, jsonify
from scraper import scrape_website
from rag_pipeline import RAGPipeline

app = Flask(__name__)

# Load Groq API key from environment
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")


# Home Page

@app.route("/")
def index():
    return render_template("index.html")



# Build Knowledge Base

@app.route("/build", methods=["POST"])
def build():
    data = request.json
    url = data.get("url")
    depth = int(data.get("depth", 3))

    if not url:
        return jsonify({"error": "URL is required."})

    try:
        # Scrape website
        content = scrape_website(url, max_pages=depth)

        if not content.strip():
            return jsonify({"error": "No content scraped from website."})

        # Create and store pipeline inside Flask app
        app.pipeline = RAGPipeline(groq_api_key=groq_api_key)
        app.pipeline.build_knowledge_base(content)

        return jsonify({"message": "Knowledge base built successfully!"})

    except Exception as e:
        return jsonify({"error": str(e)})



# Ask Question

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "Question is required."})

    # Check if knowledge base exists
    if not hasattr(app, "pipeline"):
        return jsonify({"error": "Knowledge base not built yet."})

    try:
        answer = app.pipeline.generate_answer(question)
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)})



# Run App

if __name__ == "__main__":
    app.run(debug=False)
