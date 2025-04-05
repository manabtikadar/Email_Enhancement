from flask import Flask, redirect, request, session, jsonify
from Graph import compile_full_agent_graph
main_agent = compile_full_agent_graph()
from langchain_core.runnables import RunnableConfig

app = Flask(__name__)

@app.route("/")
def home():
    print(" / route hit")
    return " Backend working"

@app.route("/query", methods=["POST"])
def query():
    print(" Querying main agent")
    data = request.json
    print(data)
    # input_data = data.get("input")
    # print(input_data)
    # if not input_data:
    #     return jsonify({"error": "No input provided"}), 400

    # Define configuration
    config = RunnableConfig(
        configurable={"user_id": "user_12345"}
    )
    # Run the main agent with the provided input
    result = main_agent.invoke(data, config=config)
    
    # Return the result as JSON
    return jsonify({"result": result})

if __name__ == "__main__":
    print(" Starting Flask backend on http://localhost:5000")
    app.run(port=5000, debug=True)