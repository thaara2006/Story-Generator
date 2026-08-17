import os
import re
import json
import urllib.parse

import boto3
import requests
import streamlit as st

from dotenv import load_dotenv
from gtts import gTTS


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Optional Pollinations API key
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY")


# ============================================================
# AWS CLIENT
# ============================================================

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AI Story Generator",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Story Generator")
st.caption(
    "Detailed Story + AI Illustrations + Narration"
)


# ============================================================
# USER INPUT
# ============================================================

topic = st.text_input(
    "Enter Story Topic",
    placeholder="Example: A boy travels through space"
)

image_style = st.selectbox(
    "Choose Image Style",
    [
        "Watercolor",
        "Sketch",
        "Cartoon",
        "Storybook"
    ]
)

language = st.selectbox(
    "Choose Narration Language",
    [
        "English",
        "Tamil",
        "Hindi"
    ]
)


# ============================================================
# AWS NOVA
# ============================================================

def ask_nova(prompt, tokens=1500, temperature=0.7):

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "inferenceConfig": {
            "max_new_tokens": tokens,
            "temperature": temperature
        }
    }

    response = bedrock.invoke_model(
        modelId="amazon.nova-lite-v1:0",
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(
        response["body"].read()
    )

    return result["output"]["message"]["content"][0]["text"]


# ============================================================
# STORY GENERATION
# ============================================================

def generate_story(topic):

    prompt = f"""
Write a detailed and creative story about:

{topic}

Requirements:

1. Write 7 to 10 natural paragraphs.
2. Do NOT number the paragraphs.
3. Do NOT use Scene 1, Scene 2, etc.
4. Each paragraph should continue the previous paragraph.
5. The story must have:
   - introduction
   - character development
   - exploration
   - conflict
   - rising action
   - climax
   - resolution
   - emotional ending
6. Use vivid descriptions.
7. Make the story suitable for a storybook.
8. Keep the same main characters throughout the story.
9. Make every paragraph visually meaningful so it can later
   be converted into an illustration.
10. Do not add explanations before or after the story.

Return ONLY the story.
"""

    return ask_nova(
        prompt,
        tokens=2200,
        temperature=0.8
    )


# ============================================================
# SPLIT STORY INTO MODULES
# ============================================================

def split_modules(story):

    paragraphs = re.split(
        r"\n\s*\n",
        story.strip()
    )

    paragraphs = [
        p.strip()
        for p in paragraphs
        if len(p.strip()) > 40
    ]

    return paragraphs


# ============================================================
# CHARACTER BIBLE
# ============================================================

def create_character_bible(story):

    prompt = f"""
Analyze this story and create a short visual character bible.

Story:
{story}

For every important recurring character provide:

Name:
Age:
Gender:
Hair:
Clothing:
Body type:
Important visual features:

Keep the descriptions simple and consistent.

Return only the character descriptions.
"""

    return ask_nova(
        prompt,
        tokens=800,
        temperature=0.3
    )


# ============================================================
# VISUAL SCENE PLANNER
# ============================================================

def create_visual_prompt(
    paragraph,
    character_bible,
    style
):

    prompt = f"""
You are a professional children's storybook illustrator.

Create ONE highly detailed image prompt from the paragraph below.

CHARACTER BIBLE:
{character_bible}

STORY PARAGRAPH:
{paragraph}

ART STYLE:
{style}

The image must be a meaningful illustration.

Describe:

- main character
- exact action
- facial expression
- emotion
- environment
- important objects
- foreground
- middle ground
- background
- lighting
- camera/viewpoint
- spatial relationships between objects

Important:

The character must have normal human anatomy.

Objects must have realistic relationships with each other.

Do NOT create fantasy creatures unless the story explicitly requires them.

Do NOT randomly add objects.

Do NOT make the image photorealistic.

The result should look like a carefully illustrated children's
storybook painting.

Return ONLY the image prompt.
"""

    return ask_nova(
        prompt,
        tokens=900,
        temperature=0.5
    )


# ============================================================
# STYLE
# ============================================================

def get_style(style):

    styles = {

        "Watercolor":
        """
        traditional watercolor children's book illustration,
        hand-painted paper texture,
        soft watercolor washes,
        delicate ink outlines,
        natural human anatomy,
        expressive faces,
        harmonious pastel colors,
        carefully composed illustration
        """,

        "Sketch":
        """
        hand-drawn pencil and ink children's book illustration,
        graphite lines,
        subtle shading,
        paper texture,
        clean anatomy,
        expressive characters,
        carefully composed drawing
        """,

        "Cartoon":
        """
        high-quality children's cartoon illustration,
        clean hand-drawn outlines,
        expressive faces,
        natural anatomy,
        soft colors,
        storybook composition,
        polished 2D illustration
        """,

        "Storybook":
        """
        classic children's storybook illustration,
        hand-painted appearance,
        watercolor and ink,
        soft paper texture,
        expressive characters,
        beautiful composition,
        warm artistic atmosphere
        """
    }

    return styles.get(
        style,
        styles["Watercolor"]
    )


# ============================================================
# IMAGE GENERATION
# ============================================================

def generate_image(
    visual_prompt,
    style
):

    style_description = get_style(style)

    negative_prompt = """
photorealistic,
3d render,
uncanny,
deformed human,
extra arms,
extra legs,
extra fingers,
missing fingers,
duplicate character,
multiple heads,
merged bodies,
floating objects,
random objects,
distorted face,
bad anatomy,
mutated hands,
disconnected limbs,
unreal species,
alien anatomy,
blurry,
low quality,
text,
letters,
watermark
"""

    final_prompt = f"""
{style_description}

{visual_prompt}

IMPORTANT:
Create one coherent illustration.
Keep every character anatomically normal.
Place objects according to the described foreground,
middle ground and background.
Make the scene look intentionally painted by a human
storybook artist.
"""

    encoded_prompt = urllib.parse.quote(
        final_prompt
    )

    url = (
        "https://gen.pollinations.ai/image/"
        + encoded_prompt
        + "?model=flux"
        + "&width=768"
        + "&height=512"
        + "&nologo=true"
    )

    headers = {}

    if POLLINATIONS_API_KEY:
        headers["Authorization"] = (
            f"Bearer {POLLINATIONS_API_KEY}"
        )

    response = requests.get(
        url,
        headers=headers,
        timeout=180
    )

    if response.status_code != 200:

        raise Exception(
            f"Image API failed: "
            f"{response.status_code} "
            f"{response.text[:300]}"
        )

    return response.content


# ============================================================
# CAPTION
# ============================================================

def create_caption(paragraph):

    prompt = f"""
Create two short beautiful lines that summarize this
story paragraph as a visual caption.

Paragraph:
{paragraph}

Rules:

- Exactly 2 lines.
- Emotional.
- Simple.
- Storybook style.
- Do not number the lines.
"""

    return ask_nova(
        prompt,
        tokens=150,
        temperature=0.5
    )


# ============================================================
# TRANSLATE FOR NARRATION
# ============================================================

def translate_story(story, language):

    if language == "English":
        return story

    prompt = f"""
Translate the following story into {language}.

Preserve:
- meaning
- emotions
- character names
- paragraph structure

Return only the translated story.

Story:
{story}
"""

    return ask_nova(
        prompt,
        tokens=2500,
        temperature=0.3
    )


# ============================================================
# AUDIO
# ============================================================

def generate_audio(text, language):

    language_codes = {
        "English": "en",
        "Tamil": "ta",
        "Hindi": "hi"
    }

    code = language_codes[language]

    tts = gTTS(
        text=text[:5000],
        lang=code,
        slow=False
    )

    audio_file = "story_narration.mp3"

    tts.save(audio_file)

    return audio_file


# ============================================================
# MAIN
# ============================================================

if st.button(
    "✨ Generate Story",
    use_container_width=True
):

    if not topic.strip():

        st.warning(
            "Please enter a story topic."
        )

        st.stop()


    # --------------------------------------------------------
    # STORY
    # --------------------------------------------------------

    with st.spinner(
        "✍️ Creating your story..."
    ):

        try:

            story = generate_story(topic)

        except Exception as e:

            st.error(
                f"Story generation failed: {e}"
            )

            st.stop()


    st.subheader("📖 Full Story")

    st.write(story)


    # --------------------------------------------------------
    # CHARACTER BIBLE
    # --------------------------------------------------------

    with st.spinner(
        "🎭 Understanding characters..."
    ):

        try:

            character_bible = (
                create_character_bible(story)
            )

        except Exception:

            character_bible = (
                "Use consistent characters "
                "throughout the illustrations."
            )


    # --------------------------------------------------------
    # MODULES
    # --------------------------------------------------------

    modules = split_modules(story)

    st.subheader(
        f"🎨 Story Modules ({len(modules)})"
    )


    # --------------------------------------------------------
    # IMAGES
    # --------------------------------------------------------

    for index, module in enumerate(
        modules,
        start=1
    ):

        st.markdown(
            f"## Module {index}"
        )

        # Visual planning
        with st.spinner(
            f"🎨 Planning illustration {index}..."
        ):

            try:

                visual_prompt = create_visual_prompt(
                    module,
                    character_bible,
                    image_style
                )

            except Exception as e:

                st.error(
                    f"Visual planning failed: {e}"
                )

                visual_prompt = module


        # Image
        with st.spinner(
            f"🖌️ Painting illustration {index}..."
        ):

            try:

                image = generate_image(
                    visual_prompt,
                    image_style
                )

                st.image(
                    image,
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"Image generation failed: {e}"
                )


        # Caption
        with st.spinner(
            "✨ Creating caption..."
        ):

            try:

                caption = create_caption(
                    module
                )

                st.markdown(
                    f"✨ {caption}"
                )

            except Exception:

                pass


        # Paragraph
        with st.expander(
            "📖 Read Full Paragraph"
        ):

            st.write(module)


    # --------------------------------------------------------
    # NARRATION
    # --------------------------------------------------------

    st.subheader(
        "🔊 Narration"
    )

    with st.spinner(
        "🎙️ Creating narration..."
    ):

        try:

            narration_text = translate_story(
                story,
                language
            )

            audio_file = generate_audio(
                narration_text,
                language
            )

            st.audio(
                audio_file,
                format="audio/mp3"
            )

        except Exception as e:

            st.error(
                f"Narration generation failed: {e}"
            )