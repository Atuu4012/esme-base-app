
import os
from langfuse import observe, get_client
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # Load environment variables from .env file

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

langfuse = get_client(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY")
)


@observe()
def calculate(operation: str, num1: float, num2: float) -> float:
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        if num2 != 0:
            return num1 / num2
        else:
            raise ValueError("Cannot divide by zero.")
    else:
        raise ValueError(f"Unsupported operation: {operation}")
    
@observe()
def get_log(number: float) -> float:
    import math
    if number > 0:
        return math.log10(number)
    else:
        raise ValueError("Number must be greater than zero to calculate log 10.")

@observe()
def get_polynomial_roots(coefficients: list[float]) -> list[complex]:
    try:
        import numpy as np

        return np.roots(coefficients).tolist()
    except Exception:
        # fallback: handle quadratic polynomials if numpy is not available
        if len(coefficients) == 3:
            a, b, c = coefficients
            import cmath

            disc = b * b - 4 * a * c
            r1 = (-b + cmath.sqrt(disc)) / (2 * a)
            r2 = (-b - cmath.sqrt(disc)) / (2 * a)
            return [r1, r2]
        raise

@observe()
def get_weather(location: str) -> str:
    
    fake_data = {
        "New York": "Sunny, 25°C",
        "San Francisco": "Foggy, 15°C",
        "London": "Rainy, 10°C",
        "Paris": "Cloudy, 20°C"
    }
    return fake_data.get(location, "Weather data not available for this location.")

@observe()
def get_stock_quantity(product_id: str) -> int:
    fake_stock_data = {
        "1": 100,
        "2": 50,
        "3": 0,
        "4": 200,
        "5": 75
    }
    return fake_stock_data.get(product_id, "Stock data not available for this product.")





# declaration avec smolagents :

from smolagents import Tool, CodeAgent, LiteLLMModel


class CalculateTool(Tool):
    name = "calculate"
    description = "A calculator that can perform basic arithmetic operations."
    inputs = {
        "operation": {"type": "string", "description": "add, subtract, multiply, divide"},
        "num1": {"type": "number", "description": "First number"},
        "num2": {"type": "number", "description": "Second number"},
    }
    output_type = "number"

    def forward(self, operation: str, num1: float, num2: float):
        return calculate(operation, num1, num2)


class GetLogTool(Tool):
    name = "get_log"
    description = "Compute base-10 logarithm of a positive number."
    inputs = {"number": {"type": "number", "description": "Input number"}}
    output_type = "number"

    def forward(self, number: float):
        return get_log(number)


class PolynomialRootsTool(Tool):
    name = "get_polynomial_roots"
    description = "Return polynomial roots from coefficients (highest degree first)."
    inputs = {"coefficients": {"type": "array", "description": "List of coefficients"}}
    output_type = "array"

    def forward(self, coefficients: list[float]):
        return get_polynomial_roots(coefficients)


class WeatherTool(Tool):
    name = "get_weather"
    description = "Return fake current weather for a few predefined locations."
    inputs = {"location": {"type": "string", "description": "Location name"}}
    output_type = "string"

    def forward(self, location: str):
        return get_weather(location)


class StockTool(Tool):
    name = "get_stock_quantity"
    description = "Return fake stock quantity for a product id."
    inputs = {"product_id": {"type": "string", "description": "Product id"}}
    output_type = "integer"

    def forward(self, product_id: str):
        return get_stock_quantity(product_id)


tools = [
    CalculateTool(),
    GetLogTool(),
    PolynomialRootsTool(),
    WeatherTool(),
    StockTool(),
]


model = LiteLLMModel(
        model_id=os.getenv("LITELLM_MODEL", "groq/llama-3.3-70b-versatile"),
        temperature=0.5,
        max_tokens=4096,
    )


agent = CodeAgent(
    tools=tools,
    model=model,
    stream_outputs=True,
    max_steps=10,
    verbosity_level=1,
)


if __name__ == "__main__": 
    with langfuse.start_as_current_observation(
        name="agent_execution", 
        as_type="span"
    ) as span : 
        
        response = agent.run(
        """
        Quel temps fait-il à Paris et quelle est la racine du polynôme x^4 + x^2 - 5x + 6 ?
        Donne-moi la réponse sous forme de JSON avec les clés "weather" et "roots"
        """
        )
        span.update(output={"response": response})

    print(response)
