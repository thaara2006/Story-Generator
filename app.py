import streamlit as st
import boto3
import json
import os
import re
from dotenv import load_dotenv

from diffusers import StableDiffusionPipeline
import torch
from gtts import gTTS


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


# ============================================================
# AWS BEDROCK CLIENT
# ============================================================

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Story Generator",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Story Generator")
st.caption("Detailed Story + Meaningful AI Illustrations + Narration")


# ============================================================
# USER INPUT
# ============================================================

topic = st.text_input(
    "Enter Story Topic",
    placeholder="Example: A boy travels through space"
)

style = st.selectbox(
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
# LOAD LOCAL STABLE DIFFUSION MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not torch.cuda.is_available():
        st.error(
            "CUDA GPU is not available. "
            "Please check your NVIDIA GPU and PyTorch installation."
        )
        st.stop()

    # Existing local model downloaded on this computer.
    # local_files_only=True prevents Hugging Face 429/download errors.
    model_path = (
        r"D:\hf_cache\hub\models--runwayml--stable-diffusion-v1-5"
        r"\snapshots\451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        local_files_only=True
    )

    # Optimizations for RTX 3050 4 GB
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    return pipe


pipe = load_model()


# ============================================================
# BEDROCK FUNCTION
# ============================================================

def ask_bedrock(prompt, tokens=1500, temperature=0.8):

    body = json.dumps({
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
    })

    response = bedrock.invoke_model(
        modelId="amazon.nova-lite-v1:0",
        body=body,
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())

    return result["output"]["message"]["content"][0]["text"]


# ============================================================
# STORY GENERATION
# ============================================================

def generate_story(topic):

    prompt = f"""
Create a detailed, original and emotionally engaging story
based on this topic:

{topic}

STORY REQUIREMENTS:

1. Do NOT force the story into exactly 5 scenes.
2. Generate 6 to 10 meaningful paragraphs/modules,
   depending on how naturally the story develops.
3. Each paragraph must represent ONE important visual
   story moment that can be illustrated as a single image.
4. Do not put several major actions into one paragraph.
5. Each paragraph should have:
   - one main action
   - one clear setting
   - one emotional moment
   - only a few important objects
6. Include:
   - introduction
   - character development
   - journey
   - challenge/conflict
   - rising action
   - climax
   - resolution
   - meaningful ending
7. Make the story detailed, emotional and cinematic.
8. Maintain consistency in character names, appearance,
   locations and important objects.
9. Use normal paragraphs only.
10. Do NOT use scene numbers.
11. Do NOT use module headings.
12. Do NOT use bullet points.
13. Do NOT add explanations.
14. Return ONLY the complete story.
"""

    return ask_bedrock(
        prompt,
        tokens=2200,
        temperature=0.8
    )


# ============================================================
# CHARACTER BIBLE
# Creates a stable visual description for recurring characters.
# ============================================================

def create_character_bible(story):

    prompt = f"""
You are a professional children's storybook character designer.

Read the story below and create a CONSISTENT visual character
description for the main recurring character(s).

STORY:
{story}

For each important recurring character, describe:
- name
- approximate age
- gender if clearly established
- hair
- eyes
- face
- clothing
- shoes
- important accessories
- distinctive visual features

Rules:
- Keep descriptions realistic and human unless the story
  explicitly requires a non-human character.
- Do not invent unnecessary characters.
- Do not change the character's appearance between paragraphs.
- Keep the descriptions concise.
- Return only the character descriptions.
"""

    return ask_bedrock(
        prompt,
        tokens=450,
        temperature=0.2
    )


# ============================================================
# SPLIT STORY INTO VISUAL MODULES
# ============================================================

def split_modules(story):

    # Remove accidental markdown heading markers.
    story = re.sub(r"^\s*#+\s*", "", story, flags=re.MULTILINE)

    # First try paragraph separation.
    modules = [
        p.strip()
        for p in re.split(r"\n\s*\n+", story)
        if p.strip()
    ]

    # Fallback if the model returned one-line paragraphs.
    if len(modules) < 2:
        modules = [
            p.strip()
            for p in story.split("\n")
            if p.strip()
        ]

    # Keep each module as one meaningful story unit.
    return modules


# ============================================================
# VISUAL SCENE PLANNER
# Converts a story paragraph into a concrete image blueprint.
# ============================================================

def create_visual_prompt(module_text, character_bible, style):

    prompt = f"""
You are a professional children's storybook illustrator
and visual scene composition expert.

Convert ONE story paragraph into ONE clear visual scene
for an image-generation model.

CHARACTER BIBLE:
{character_bible}

STORY PARAGRAPH:
{module_text}

REQUESTED ART STYLE:
{style}

Create a precise visual blueprint.

Include:
1. Main subject and appearance.
2. Exactly what the main character is doing.
3. Facial expression/emotion.
4. Setting and time of day.
5. Important objects that MUST appear.
6. Foreground.
7. Middle ground.
8. Background.
9. Spatial arrangement and relative positions.
10. Camera/view composition.

IMPORTANT RULES:
- Show ONE main moment only.
- Do not combine several events from the paragraph.
- Do not invent unrelated objects.
- Keep the recurring character's appearance identical
  to the character bible.
- Keep people anatomically natural.
- Clearly separate people and objects.
- Describe physical relationships between objects.
- Avoid surreal or abstract interpretations.
- Prefer a simple readable composition over a crowded image.
- The image should look like a traditional children's
  storybook illustration.
- Do not include text, letters, titles, captions or logos.

Return ONLY a detailed visual description, approximately
120-180 words.
"""

    return ask_bedrock(
        prompt,
        tokens=400,
        temperature=0.25
    )


# ============================================================
# CAPTION GENERATION
# ============================================================

def create_caption(module_text):

    prompt = f"""
Read this story paragraph:

{module_text}

Write exactly TWO short, beautiful sentences that can
be displayed below an illustration.

Requirements:
- Capture the main action and emotion.
- Do not mechanically summarize.
- Simple but expressive language.
- No numbering.
- No quotation marks.
- Return only the two sentences.
"""

    return ask_bedrock(
        prompt,
        tokens=150,
        temperature=0.5
    )


# ============================================================
# IMAGE STYLE PROMPTS
# ============================================================

def get_style_prompt(style):

    if style == "Watercolor":
        return """
traditional children's book watercolor painting,
hand-painted watercolor on textured paper,
soft natural brush strokes,
gentle color blending,
delicate ink outlines,
subtle paper texture,
warm artistic atmosphere,
clear readable composition,
beautiful traditional storybook artwork
"""

    elif style == "Sketch":
        return """
traditional pencil and ink children's book drawing,
hand-drawn graphite line art,
fine pencil shading,
natural human proportions,
delicate ink outlines,
paper texture,
clean readable composition,
traditional illustration
"""

    elif style == "Cartoon":
        return """
hand-drawn 2D children's cartoon illustration,
clean expressive linework,
soft painted colors,
natural character proportions,
expressive but believable faces,
clear readable composition,
traditional storybook cartoon artwork
"""

    return """
traditional children's storybook illustration,
hand-painted artwork on paper,
soft ink outlines,
gentle colors,
subtle paper texture,
natural character proportions,
clear visual storytelling,
beautiful children's book artwork
"""


# ============================================================
# NEGATIVE PROMPT
# Prevents common SD 1.5 image problems.
# ============================================================

NEGATIVE_PROMPT = """
photorealistic,
3d render,
cgi,
surreal,
abstract,
fantasy mutation,
unreal species,
deformed human,
distorted anatomy,
extra arms,
extra legs,
extra hands,
extra fingers,
missing fingers,
fused fingers,
duplicate person,
duplicate character,
duplicate object,
merged objects,
floating objects,
detached body parts,
twisted body,
bad proportions,
deformed face,
asymmetrical eyes,
bad hands,
malformed hands,
cluttered composition,
random objects,
unrelated objects,
confusing scene,
multiple simultaneous events,
extreme close-up,
cropped character,
blurry,
low quality,
text,
letters,
words,
title,
caption,
logo,
watermark
"""


# ============================================================
# IMAGE GENERATION
# ============================================================

def generate_image(
    module_text,
    character_bible,
    style
):

    # First let Nova understand the story visually.
    visual_description = create_visual_prompt(
        module_text,
        character_bible,
        style
    )

    style_description = get_style_prompt(style)

    # Stable Diffusion receives a visual blueprint,
    # not the original story paragraph.
    image_prompt = f"""
{style_description}

CHARACTER CONSISTENCY:
{character_bible}

VISUAL SCENE:
{visual_description}

COMPOSITION REQUIREMENTS:
- One clear story moment.
- Main character is clearly visible.
- Natural human anatomy.
- Objects have correct physical relationships.
- Foreground, middle ground and background are distinct.
- Balanced composition.
- The main action is visually obvious.
- Traditional painted/drawn appearance.
- No text or writing anywhere in the image.
"""

    image = pipe(
        prompt=image_prompt,
        negative_prompt=NEGATIVE_PROMPT,
        height=512,
        width=512,
        num_inference_steps=25,
        guidance_scale=6.5
    ).images[0]

    return image


# ============================================================
# LANGUAGE CODE FOR gTTS
# ============================================================

def get_language_code(language):

    if language == "Tamil":
        return "ta"

    if language == "Hindi":
        return "hi"

    return "en"


# ============================================================
# NARRATION
# ============================================================

def generate_audio(text, language):

    language_code = get_language_code(language)

    audio_file = "story_narration.mp3"

    tts = gTTS(
        text=text[:5000],
        lang=language_code,
        slow=False
    )

    tts.save(audio_file)

    return audio_file


# ============================================================
# MAIN APPLICATION
# ============================================================

if st.button(
    "✨ Generate Story",
    use_container_width=True
):

    if not topic.strip():
        st.warning("Please enter a story topic.")
        st.stop()

    # --------------------------------------------------------
    # STORY
    # --------------------------------------------------------

    with st.spinner("📖 Creating your detailed story..."):

        try:
            story = generate_story(topic)

        except Exception as e:
            st.error(f"Story generation failed:\n\n{e}")
            st.stop()

    st.subheader("📖 Full Story")
    st.write(story)

    # --------------------------------------------------------
    # CHARACTER BIBLE
    # --------------------------------------------------------

    with st.spinner("👤 Creating consistent character design..."):

        try:
            character_bible = create_character_bible(story)

        except Exception as e:
            character_bible = ""
            st.warning(
                f"Character design generation failed. "
                f"Images will still be generated.\n\n{e}"
            )

    # --------------------------------------------------------
    # STORY MODULES
    # --------------------------------------------------------

    modules = split_modules(story)

    st.subheader(
        f"🎨 Story Modules ({len(modules)})"
    )

    # --------------------------------------------------------
    # PROCESS EACH MODULE
    # --------------------------------------------------------

    for index, module in enumerate(
        modules,
        start=1
    ):

        st.markdown(
            f"## 📍 Module {index}"
        )

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        with st.spinner(
            f"🎨 Planning and painting Module {index}..."
        ):

            try:

                image = generate_image(
                    module,
                    character_bible,
                    style
                )

                st.image(
                    image,
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"Image generation failed for Module "
                    f"{index}: {e}"
                )

        # ----------------------------------------------------
        # CAPTION
        # ----------------------------------------------------

        with st.spinner(
            "✍️ Creating illustration caption..."
        ):

            try:

                caption = create_caption(
                    module
                )

                st.markdown(
                    f"### ✨ {caption}"
                )

            except Exception as e:

                st.warning(
                    f"Caption generation failed: {e}"
                )

        # ----------------------------------------------------
        # FULL MODULE
        # ----------------------------------------------------

        with st.expander(
            f"📖 Read Module {index}"
        ):

            st.write(module)

        st.markdown("---")

    # --------------------------------------------------------
    # NARRATION
    # --------------------------------------------------------

    st.subheader("🔊 Narration")

    with st.spinner(
        "🎙️ Creating narration..."
    ):

        try:

            audio_file = generate_audio(
                story,
                language
            )

            st.audio(
                audio_file,
                format="audio/mp3"
            )

            st.success(
                "Narration generated successfully!"
            )

        except Exception as e:

            st.error(
                f"Narration generation failed: {e}"
            )