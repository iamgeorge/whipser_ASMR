from flask import Flask, request, jsonify, render_template
import openai
from flask import Response
from flask_cors import CORS
import os
import uuid
import torchaudio
from audiocraft.models import musicgen

app = Flask(__name__)
CORS(app)
# Load model once globally

os.makedirs("static/music", exist_ok=True)

openai.api_key = "real key"

# Web UI for browser
@app.route('/', methods=['GET', 'POST'])
def index():
    image_url = None
    if request.method == 'POST':
        # Get form input
        name = request.form.get('name')
        location = request.form.get('location')
        purpose = request.form.get('purpose')
        looks = request.form.get('looks')
        trait = request.form.get('trait')
        outfit = request.form.get('outfit')

        # Combine into base character prompt
        raw_character_prompt = f"The character is a {name}. It is from {location}. It is designed for the purpose of being a {purpose}. It has {looks}. It is {trait}. It is wearing {outfit}."

        try:
            # Use ChatGPT to improve prompt
            chat_response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to create high quality image generation prompts for a single character design. Create a character image generation prompt for an AI text to image generator such as DALL·E from the users description. Keep it detailed and child friendly."},
                    {"role": "user", "content": raw_character_prompt}
                ]
            )
            refined_prompt = chat_response.choices[0].message.content  # Extract the refined prompt from the response

            # Generate image using DALL·E
            image_response = openai.images.generate(
                model="dall-e-3",  # Use the latest DALL·E model
                prompt=refined_prompt,  # Use the refined prompt from ChatGPT
                size="1024x1024",  # Specify the size of the image
                quality="standard",  # Use standard quality for the image
                style="vivid",  # Use vivid style for the image
                n=1,  # Generate a single image
            )
            image_url = image_response.data[0].url  # Extract the image URL from the response

        except Exception as e:
            print("Error:", e)

    return render_template('index.html', image_url=image_url)

# API for Postman or programmatic access
@app.route('/api/generate-character-image', methods=['POST'])
def generate_character_image():
    try:
        data = request.form
        name = data.get('name', '')
        location = data.get('location', '')
        purpose = data.get('purpose', '')
        looks = data.get('looks', '')
        trait = data.get('trait', '')
        outfit = data.get('outfit', '')

        raw_prompt = f"The character is a {name}. It is from {location}. It is designed for the purpose of being a {purpose}. It has {looks}. It is {trait}. It is wearing {outfit}."

        chat_response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to create high quality image generation prompts for a single character design. Create a character image generation prompt for an AI text to image generator such as DALL·E from the users description. Keep it detailed and child friendly."},
                {"role": "user", "content": raw_prompt}
            ]
        )
        refined_prompt = chat_response.choices[0].message.content

        image_response = openai.images.generate(
            model="dall-e-3",
            prompt=refined_prompt,
            size="1024x1024",
            quality="standard",
            style="vivid",
            n=1
        )
        image_url = image_response.data[0].url

        return Response(image_url, status=200, mimetype='text/plain')

    except Exception as e:
        print("Error:", e)
        return "Error generating image", 500
    
@app.route('/api/generate-background-image', methods=['POST'])
def generate_background_image():
    try:
        data = request.get_json()
        scene = data.get('scene', '')
        mood = data.get('mood', '')
        lighting = data.get('lighting', '')
        details = data.get('details', '')

        # Build a raw background prompt
        raw_prompt = f"A {mood} {scene} background, illuminated by {lighting}. {details}"

        # Use GPT to refine the background prompt
        chat_response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a prompt engineer helping users generate beautiful AI background scenes for ASMR and study environments. The prompts should focus on atmospheric detail, lighting, and calm mood. Keep it vivid, gentle, and realistic.Only one image in one frame, no collage or multiple images."},
                {"role": "user", "content": raw_prompt}
            ]
        )
        refined_prompt = chat_response.choices[0].message.content

        # Generate image with DALL·E
        image_response = openai.images.generate(
            model="dall-e-3",
            prompt=refined_prompt,
            size="1024x1024",
            quality="standard",
            style="vivid",
            n=1
        )
        image_url = image_response.data[0].url

        return Response(image_url, status=200, mimetype='text/plain')

    except Exception as e:
        print("Error:", e)
        return "Error generating background image", 500
    
@app.route('/api/generate-background-music', methods=['POST'])
def generate_background_music():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()

        if not prompt:
            return "Missing prompt", 400

        print("Generating music for prompt:", prompt)
        musicgen_model = musicgen.MusicGen.get_pretrained('small', device='cuda')
        musicgen_model.set_generation_params(duration=3)
        output = musicgen_model.generate([prompt], progress=True)

        # Save audio as .wav file
        filename = f"{uuid.uuid4().hex}.wav"
        filepath = os.path.join("static/music", filename)
        torchaudio.save(filepath, output[0].cpu(), 32000)

        # Return file URL
        file_url = f"/static/music/{filename}"
        return Response(file_url, status=200, mimetype='text/plain')

    except Exception as e:
        print("Music Generation Error:", e)
        return "Error generating music", 500



if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000, debug = True)
