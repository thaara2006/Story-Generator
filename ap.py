import streamlit as st
import boto3
import json
import os
import re
import io
import tempfile
import subprocess
import hashlib
from pathlib import Path

from dotenv import load_dotenv

from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler
)

import torch

from gtts import gTTS

from docx import Document
from docx.shared import Inches

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage
)
from reportlab.lib.units import inch

from PIL import Image

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Story Generator",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Optional:
# You can define SD_MODEL_PATH in .env if needed.
#
# Example:
# SD_MODEL_PATH=D:\hf_cache\hub\models--runwayml--stable-diffusion-v1-5\snapshots\451f4fe16113bff5a5d2269ed5ad43b0592e9a14


# ============================================================
# AWS CLIENTS
# ============================================================

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

polly = boto3.client(
    "polly",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "story_generated": False,

    "story": "",
    "topic": "",
    "style": "Watercolor",
    "language": "English",

    "modules": [],
    "character_bible": "",

    # IMPORTANT:
    # One image slot corresponds to one module.
    # Failed image = None.
    "generated_images": [],

    "generated_captions": [],

    # One audio slot corresponds to one module.
    "module_audio": [],
    "module_durations": [],

    "full_audio": None,
    "audio_source": "",

    "pdf_bytes": None,
    "word_bytes": None,
    "video_bytes": None
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# TITLE
# ============================================================

st.title("📚 AI Story Generator")

st.caption(
    "AI Story + Meaningful Scene Illustrations + Full Narration + Story Video"
)


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
# FIND LOCAL MODEL
# ============================================================

def find_local_model():

    # --------------------------------------------------------
    # 1. Check .env first
    # --------------------------------------------------------

    env_model_path = os.getenv(
        "SD_MODEL_PATH"
    )

    if env_model_path:

        env_model_path = os.path.abspath(
            os.path.expandvars(
                os.path.expanduser(
                    env_model_path
                )
            )
        )

        if os.path.exists(env_model_path):

            return env_model_path


    # --------------------------------------------------------
    # 2. Check your existing local cache
    # --------------------------------------------------------

    cache_root = Path(
        r"D:\hf_cache\hub\models--runwayml--stable-diffusion-v1-5\snapshots"
    )

    if not cache_root.exists():

        return None


    # --------------------------------------------------------
    # 3. Find snapshots containing model_index.json
    # --------------------------------------------------------

    snapshots = []

    for folder in cache_root.iterdir():

        if not folder.is_dir():
            continue

        model_index = (
            folder /
            "model_index.json"
        )

        if model_index.exists():

            snapshots.append(
                folder
            )


    if not snapshots:

        return None


    # --------------------------------------------------------
    # 4. Use latest snapshot
    # --------------------------------------------------------

    snapshots.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return str(
        snapshots[0]
    )


# ============================================================
# LOAD LOCAL STABLE DIFFUSION
# ============================================================

@st.cache_resource
def load_model():

    # --------------------------------------------------------
    # CUDA CHECK
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        st.error(
            "❌ CUDA GPU is not available.\n\n"
            "Your RTX GPU must be detected by PyTorch."
        )

        st.stop()


    # --------------------------------------------------------
    # GET LOCAL MODEL PATH
    # --------------------------------------------------------

    model_path = find_local_model()


    if model_path is None:

        st.error(
            "❌ Stable Diffusion model path was not found.\n\n"
            "Please check your local Hugging Face cache."
        )

        st.stop()


    # --------------------------------------------------------
    # SHOW MODEL PATH
    # --------------------------------------------------------

    st.info(
        "🎨 Local Stable Diffusion model found:\n\n"
        + model_path
    )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    with st.spinner(
        "🎨 Loading local Stable Diffusion model..."
    ):

        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=True,
            safety_checker=None
        )


    # --------------------------------------------------------
    # DPMSOLVER SCHEDULER
    # --------------------------------------------------------

    pipe.scheduler = (
        DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
    )


    # --------------------------------------------------------
    # GPU MEMORY OPTIMIZATION
    # --------------------------------------------------------

    pipe.enable_attention_slicing()

    pipe.enable_vae_slicing()

    pipe.enable_model_cpu_offload()


    # --------------------------------------------------------
    # CUDA OPTIMIZATION
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.backends.cuda.matmul.allow_tf32 = True


    return pipe


# ============================================================
# LOAD MODEL ONLY ONCE
# ============================================================

pipe = load_model()


# ============================================================
# BEDROCK TEXT GENERATION
# ============================================================

def ask_bedrock(
    prompt,
    tokens=1500,
    temperature=0.8
):

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


    result = json.loads(
        response["body"].read()
    )


    return (
        result[
            "output"
        ][
            "message"
        ][
            "content"
        ][0]["text"]
    )


# ============================================================
# STORY GENERATION
# ============================================================

def generate_story(topic):

    prompt = f"""
Create an original and emotionally engaging story based on:

{topic}

Generate 6 to 10 natural story modules.

Each module must describe ONE clearly visible moment.

Each module must contain:

- Who is present
- Where they are
- What they are physically doing
- What important object is involved
- What emotion is visible
- What happens immediately in that moment

IMPORTANT:

Every module must be easy to convert into ONE illustration.

Do NOT combine several major actions into one image.

For example:

BAD:
"The boy enters the room, discovers a book,
reads it and suddenly travels to another world."

GOOD:
"The boy stands inside the old library and notices
a glowing book on a wooden table."

Then another module can show him opening the book.

The complete story should naturally contain:

Beginning
Character development
Journey
Challenge
Rising action
Climax
Resolution
Meaningful ending

Maintain consistency in:

- Character names
- Age
- Hair
- Clothing
- Important objects
- Locations

Do not unnecessarily assume:

- Nationality
- Religion
- Culture
- Traditional clothing
- Festivals
- Architecture
- Food
- Cultural symbols

unless explicitly required by the story.

Characters should be ordinary believable human beings.

Do not make the story dependent on one country's culture.

Use normal paragraphs.

No headings.

No numbering.

No bullet points.

No explanations.

Return ONLY the complete story.
"""


    return ask_bedrock(
        prompt,
        tokens=2400,
        temperature=0.8
    )


# ============================================================
# CHARACTER BIBLE
# ============================================================

def create_character_bible(story):

    prompt = f"""
You are designing consistent 2D storybook characters.

STORY:

{story}

Identify only the important recurring characters.

For every recurring character provide:

NAME:
AGE:
GENDER:
HAIR STYLE:
HAIR COLOR:
EYE COLOR:
SKIN TONE:
FACE:
BODY TYPE:
CLOTHING:
SHOES:
ACCESSORIES:
DISTINCTIVE FEATURES:

RULES:

- Normal human appearance.
- 2D illustrated appearance.
- No photorealism.
- No fantasy creatures unless explicitly required.
- Keep the same appearance in every scene.
- Keep clothing consistent unless the story changes it.
- Do not add cultural clothing unless explicitly required.
- Do not add religious symbols.
- Do not invent unnecessary characters.
- Keep descriptions concise.

Return only the character information.
"""


    return ask_bedrock(
        prompt,
        tokens=500,
        temperature=0.1
    )


# ============================================================
# SPLIT STORY
# ============================================================

def split_modules(story):

    story = re.sub(
        r"^\s*#+\s*",
        "",
        story,
        flags=re.MULTILINE
    )


    modules = [

        p.strip()

        for p in re.split(
            r"\n\s*\n+",
            story
        )

        if p.strip()

    ]


    if len(modules) < 2:

        modules = [

            p.strip()

            for p in story.split("\n")

            if p.strip()

        ]


    return modules


# ============================================================
# IMAGE STYLE
# ============================================================

def get_style_prompt(style):

    base = """
2D hand-drawn children's storybook illustration,
clear visual storytelling,
ordinary human characters,
natural human anatomy,
natural body proportions,
clean controlled outlines,
clear foreground,
clear middle ground,
clear background,
simple uncluttered composition,
expressive faces,
clear body posture,
meaningful physical interaction,
traditional illustration,
not photorealistic,
not 3D,
not CGI,
not a photograph
"""


    if style == "Watercolor":

        return base + """

traditional children's book watercolor,
hand-painted watercolor on textured paper,
soft watercolor washes,
gentle brush strokes,
delicate ink outlines,
subtle paper texture,
warm artistic appearance,
beautiful hand-painted illustration
"""


    if style == "Sketch":

        return """
clean 2D children's storybook sketch,
professional hand-drawn pencil and ink illustration,
clean smooth black outlines,
controlled linework,
clear facial structure,
two clearly visible eyes,
properly aligned eyes,
clear eyebrows,
simple defined nose,
clear mouth,
natural face shape,
natural head proportions,
natural body proportions,
clean hands,
simple expressive facial expressions,
consistent character design,
clear character silhouettes,
light pencil shading,
controlled cross-hatching,
simple readable background,
main characters clearly visible,
professional children's book sketch,
2D artwork,
not photorealistic,
not 3D,
not CGI,
not a rough doodle,
not scribbled line art
"""


    if style == "Cartoon":

        return base + """

2D children's cartoon illustration,
clean expressive outlines,
soft painted colors,
friendly natural faces,
clear eyes,
clear mouth,
simple readable shapes,
natural human proportions,
professional animated storybook appearance
"""


    return base + """

classic children's storybook illustration,
hand-painted artwork,
delicate ink outlines,
soft colors,
subtle paper texture,
natural human characters,
clear visual storytelling,
traditional illustrated book appearance
"""


# ============================================================
# NEGATIVE PROMPT
# ============================================================

NEGATIVE_PROMPT = """
photorealistic,
photo,
real photograph,
3d,
3d render,
cgi,

surreal,
abstract,
symbolic,
metaphorical,
dreamlike,

bad face,
poor face,
unclear face,
faceless,
blank face,
featureless face,
rough face,
scribbled face,
messy face,

distorted facial features,
incorrect eyes,
uneven eyes,
misaligned eyes,
cross eyed,
missing eyes,
tiny eyes,
missing nose,
missing mouth,
deformed mouth,
asymmetrical face,
elongated face,
compressed face,
disfigured face,
melting face,
blurred face,
unrecognizable face,

bad anatomy,
bad proportions,
extra arms,
extra legs,
extra hands,
extra fingers,
missing fingers,
fused fingers,
malformed hands,

duplicate person,
duplicate character,
duplicate face,
extra head,
merged bodies,

incorrect object interaction,
floating object,
merged object,
impossible physics,

random objects,
random decorations,
random buildings,
random vehicles,
random animals,

random culture,
cultural stereotype,
traditional clothing,
religious symbols,
festival decorations,

painting inside image,
picture frame,
framed painting,
open sketchbook,
drawing on paper,
photograph of artwork,

text,
letters,
words,
caption,
subtitle,
logo,
watermark,

rough doodle,
scribbles,
messy sketch,
unfinished drawing,
rough line art,

cropped character,
cropped face,
extreme closeup,
blurry,
low quality,
cluttered composition,
confusing scene,
multiple scenes
"""


# ============================================================
# VISUAL SCENE PLANNER
# ============================================================

def create_visual_prompt(
    module_text,
    character_bible
):

    prompt = f"""
You are a professional children's book storyboard artist.

Your job is to convert ONE story paragraph into ONE
physically understandable illustration.

Do NOT create a generic beautiful background.

Do NOT create a symbolic interpretation.

Do NOT add artistic objects that are not present.

STORY MODULE:

{module_text}

CHARACTER BIBLE:

{character_bible}

Analyze ONLY what can physically be shown.

Return:

CHARACTERS:
List the characters physically present.

ACTION:
Describe exactly what each character is doing.

LOCATION:
Describe the exact physical location.

IMPORTANT OBJECTS:
List only objects that matter to this moment.

CHARACTER POSITIONS:
Explain where every important character is located
relative to other characters and objects.

FACIAL EXPRESSIONS:
Describe the visible emotion.

BACKGROUND:
Describe only the necessary background.

LIGHTING:
Describe the lighting if the story supports it.

CAMERA:
Describe a medium-wide composition that shows
both the people and their environment.

VISUAL PRIORITY:
State the ONE action the viewer must understand
immediately.

STRICT RULES:

1. One moment only.
2. One main action.
3. No symbolic imagery.
4. No random objects.
5. No unnecessary characters.
6. No cultural assumptions.
7. No text.
8. No logos.
9. Characters interact physically and naturally.
10. The image must make sense without reading the story.
11. Keep the composition simple.
12. Keep recurring character appearance consistent.
13. Characters are normal human beings.
14. Use universal environments if culture is unspecified.
15. Do not turn ordinary objects into magical objects.
16. Maximum 3 clearly detailed foreground characters.
17. If additional people are needed, place them farther
    in the background as simple figures.
18. Do not create a wall of faces.
19. Do not overlap important faces.
20. Important characters must be large enough for their
    expressions to be recognizable.

Return only the requested fields.
"""


    return ask_bedrock(
        prompt,
        tokens=650,
        temperature=0.1
    )


# ============================================================
# BUILD IMAGE PROMPT
# ============================================================

def build_image_prompt(
    module_text,
    character_bible,
    visual_description,
    style
):

    style_description = get_style_prompt(
        style
    )


    return f"""
{style_description}

==================================================
EXACT STORY MOMENT
==================================================

{module_text}

==================================================
CHARACTER DESIGN
==================================================

{character_bible}

==================================================
VISUAL SCENE BLUEPRINT
==================================================

{visual_description}

==================================================
IMPORTANT VISUAL RULE
==================================================

The illustration MUST show the actual event from
the story module.

Do not create a generic scene.

Do not create a random fantasy landscape.

Do not replace an action with a symbol.

The viewer should understand the scene without
reading the paragraph.

==================================================
FACE DESIGN
==================================================

Faces should be simple, clean and readable.

Each important character should have:

- two clearly positioned eyes
- visible eyebrows
- a defined nose
- a defined mouth
- natural face shape
- symmetrical facial structure
- clear expression
- consistent head shape

Do not make faces photorealistic.

Do not make faces hyper-detailed.

Do not make faces tiny.

==================================================
BODY AND ACTION
==================================================

Show the complete body posture whenever possible.

If someone is:

walking:
show walking posture.

sitting:
show the person physically sitting.

holding:
show the hand touching the object.

opening:
show the object being opened.

reading:
show the person looking at the book/document.

talking:
show characters facing each other.

running:
show a natural running posture.

reaching:
show the arm actually reaching the object.

==================================================
COMPOSITION
==================================================

Use a medium-wide storybook composition.

Main characters:
foreground or middle ground.

Important object:
clearly visible.

Environment:
visible enough to identify the location.

Background:
simple and supportive.

Do not overcrowd the scene.

Do not create multiple scenes in one image.

No text.

No letters.

No captions.

No logos.

No watermark.
"""


# ============================================================
# MODULE SEED
# ============================================================

def get_module_seed(
    story,
    module_index
):

    value = (
        story
        +
        str(module_index)
    )


    digest = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


    return int(
        digest[:8],
        16
    )


# ============================================================
# GENERATE IMAGE
# ============================================================

def generate_image(
    module_text,
    character_bible,
    style,
    story,
    module_index
):

    # --------------------------------------------------------
    # PLAN SCENE
    # --------------------------------------------------------

    visual_description = create_visual_prompt(
        module_text,
        character_bible
    )


    # --------------------------------------------------------
    # BUILD IMAGE PROMPT
    # --------------------------------------------------------

    image_prompt = build_image_prompt(
        module_text,
        character_bible,
        visual_description,
        style
    )


    # --------------------------------------------------------
    # FIXED SEED
    # --------------------------------------------------------

    seed = get_module_seed(
        story,
        module_index
    )


    generator = torch.Generator(
        device="cuda"
    ).manual_seed(
        seed
    )


    # --------------------------------------------------------
    # GENERATE IMAGE
    # --------------------------------------------------------

    result = pipe(
        prompt=image_prompt,
        negative_prompt=NEGATIVE_PROMPT,
        height=512,
        width=512,
        num_inference_steps=30,
        guidance_scale=7.0,
        generator=generator
    )


    return result.images[0]


# ============================================================
# CAPTION
# ============================================================

def create_caption(
    module_text
):

    prompt = f"""
Write exactly two short storybook sentences
based ONLY on this module.

MODULE:

{module_text}

The sentences should describe the actual event
and emotion.

Do not add new information.

Return only two sentences.
"""


    return ask_bedrock(
        prompt,
        tokens=120,
        temperature=0.3
    )


# ============================================================
# POLLY VOICE
# ============================================================

def get_polly_voice(language):

    if language == "Tamil":

        return "Kajal", "neural"


    if language == "Hindi":

        return "Aditi", "standard"


    return "Joanna", "standard"


# ============================================================
# POLLY AUDIO
# ============================================================

def generate_polly_audio(
    text,
    language
):

    voice, engine = get_polly_voice(
        language
    )


    return polly.synthesize_speech(

        Text=text[:3000],

        OutputFormat="mp3",

        VoiceId=voice,

        Engine=engine

    )[
        "AudioStream"
    ].read()


# ============================================================
# GTTS AUDIO
# ============================================================

def generate_gtts_audio(
    text,
    language
):

    language_codes = {

        "English": "en",

        "Tamil": "ta",

        "Hindi": "hi"

    }


    lang = language_codes.get(
        language,
        "en"
    )


    file_descriptor, file_name = (
        tempfile.mkstemp(
            suffix=".mp3"
        )
    )


    os.close(
        file_descriptor
    )


    temp_file = Path(
        file_name
    )


    try:

        tts = gTTS(
            text=text[:5000],
            lang=lang,
            slow=False
        )


        tts.save(
            str(temp_file)
        )


        return temp_file.read_bytes()


    finally:

        try:
            temp_file.unlink()

        except Exception:
            pass


# ============================================================
# FFMPEG
# ============================================================

def get_ffmpeg():

    return imageio_ffmpeg.get_ffmpeg_exe()


# ============================================================
# AUDIO DURATION
# ============================================================

def get_audio_duration(
    audio_bytes
):

    ffmpeg = get_ffmpeg()


    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="audio_duration_"
        )
    )


    audio_file = (
        temp_dir /
        "audio.mp3"
    )


    try:

        audio_file.write_bytes(
            audio_bytes
        )


        command = [

            ffmpeg,

            "-i",

            str(audio_file)

        ]


        result = subprocess.run(

            command,

            capture_output=True,

            text=True

        )


        output = (
            result.stderr
            +
            result.stdout
        )


        match = re.search(

            r"Duration:\s*(\d+):(\d+):([\d.]+)",

            output

        )


        if not match:

            return 3.0


        hours = int(
            match.group(1)
        )

        minutes = int(
            match.group(2)
        )

        seconds = float(
            match.group(3)
        )


        duration = (

            hours * 3600
            +
            minutes * 60
            +
            seconds

        )


        return max(
            duration,
            2.0
        )


    finally:

        try:
            audio_file.unlink()

        except Exception:
            pass


        try:
            temp_dir.rmdir()

        except Exception:
            pass


# ============================================================
# COMBINE AUDIO FILES
# ============================================================

def combine_audio_files(
    audio_list
):

    if not audio_list:

        raise ValueError(
            "No audio files available."
        )


    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="combine_audio_"
        )
    )


    concat_file = (
        temp_dir /
        "concat.txt"
    )


    output_file = (
        temp_dir /
        "full_narration.mp3"
    )


    try:

        audio_files = []


        for index, audio in enumerate(
            audio_list
        ):

            file_path = (

                temp_dir
                /
                f"module_{index}.mp3"

            )


            file_path.write_bytes(
                audio
            )


            audio_files.append(
                file_path
            )


        with open(

            concat_file,

            "w",

            encoding="utf-8"

        ) as f:

            for audio_file in audio_files:

                path = str(
                    audio_file
                ).replace(
                    "\\",
                    "/"
                )


                f.write(
                    f"file '{path}'\n"
                )


        ffmpeg = get_ffmpeg()


        command = [

            ffmpeg,

            "-y",

            "-f",
            "concat",

            "-safe",
            "0",

            "-i",
            str(concat_file),

            "-c:a",
            "libmp3lame",

            "-b:a",
            "128k",

            str(output_file)

        ]


        result = subprocess.run(

            command,

            capture_output=True,

            text=True

        )


        if result.returncode != 0:

            raise RuntimeError(

                "Audio combination failed:\n"
                +
                result.stderr[-3000:]

            )


        return output_file.read_bytes()


    finally:

        try:
            concat_file.unlink()

        except Exception:
            pass


        try:
            output_file.unlink()

        except Exception:
            pass


        try:

            for file in temp_dir.glob("*"):

                file.unlink()

        except Exception:
            pass


        try:
            temp_dir.rmdir()

        except Exception:
            pass


# ============================================================
# FULL NARRATION
# ============================================================

def generate_full_narration(
    modules,
    language
):

    module_audio = []

    module_durations = []

    polly_success = True


    for module in modules:

        try:

            audio = generate_polly_audio(
                module,
                language
            )

        except Exception:

            polly_success = False

            audio = generate_gtts_audio(
                module,
                language
            )


        module_audio.append(
            audio
        )


        duration = get_audio_duration(
            audio
        )


        module_durations.append(
            duration
        )


    full_audio = combine_audio_files(
        module_audio
    )


    source = (
        "AWS Polly"
        if polly_success
        else "gTTS fallback"
    )


    return (
        module_audio,
        module_durations,
        full_audio,
        source
    )


# ============================================================
# WORD DOCUMENT
# ============================================================

def create_word_document(
    story,
    topic,
    modules,
    images,
    captions
):

    buffer = io.BytesIO()


    document = Document()


    document.add_heading(
        "AI Story Generator",
        0
    )


    document.add_paragraph(
        f"Story Topic: {topic}"
    )


    document.add_heading(
        "Full Story",
        level=1
    )


    for module in modules:

        document.add_paragraph(
            module
        )


    document.add_page_break()


    document.add_heading(
        "Illustrated Story",
        level=1
    )


    for index, module in enumerate(
        modules,
        start=1
    ):

        document.add_heading(
            f"Module {index}",
            level=2
        )


        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        if (
            index <= len(images)
            and
            images[index - 1] is not None
        ):

            image_buffer = io.BytesIO()


            images[
                index - 1
            ].save(

                image_buffer,

                format="PNG"

            )


            image_buffer.seek(0)


            document.add_picture(

                image_buffer,

                width=Inches(6)

            )


        # ----------------------------------------------------
        # CAPTION
        # ----------------------------------------------------

        if index <= len(captions):

            caption = captions[
                index - 1
            ]


            if caption:

                document.add_paragraph(
                    caption
                )


        # ----------------------------------------------------
        # MODULE TEXT
        # ----------------------------------------------------

        document.add_paragraph(
            module
        )


    document.save(
        buffer
    )


    return buffer.getvalue()


# ============================================================
# PDF
# ============================================================

def escape_pdf_text(
    text
):

    return (

        text
        .replace(
            "&",
            "&amp;"
        )
        .replace(
            "<",
            "&lt;"
        )
        .replace(
            ">",
            "&gt;"
        )

    )


def create_pdf_document(
    story,
    topic,
    modules,
    images,
    captions
):

    buffer = io.BytesIO()


    pdf = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=45,

        leftMargin=45,

        topMargin=45,

        bottomMargin=45

    )


    styles = getSampleStyleSheet()


    elements = []


    elements.append(

        Paragraph(

            "AI Story Generator",

            styles["Title"]

        )

    )


    elements.append(
        Spacer(1, 10)
    )


    elements.append(

        Paragraph(

            "<b>Story Topic:</b> "
            +
            escape_pdf_text(topic),

            styles["Normal"]

        )

    )


    elements.append(
        Spacer(1, 15)
    )


    elements.append(

        Paragraph(

            "Full Story",

            styles["Heading1"]

        )

    )


    for module in modules:

        elements.append(

            Paragraph(

                escape_pdf_text(module),

                styles["BodyText"]

            )

        )


        elements.append(
            Spacer(1, 10)
        )


    elements.append(

        Paragraph(

            "Illustrated Story",

            styles["Heading1"]

        )

    )


    for index, module in enumerate(
        modules,
        start=1
    ):

        elements.append(

            Paragraph(

                f"Module {index}",

                styles["Heading2"]

            )

        )


        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        if (
            index <= len(images)
            and
            images[index - 1] is not None
        ):

            image_buffer = io.BytesIO()


            images[
                index - 1
            ].save(

                image_buffer,

                format="PNG"

            )


            image_buffer.seek(0)


            elements.append(

                PDFImage(

                    image_buffer,

                    width=6.2 * inch,

                    height=4.13 * inch

                )

            )


            elements.append(
                Spacer(1, 8)
            )


        # ----------------------------------------------------
        # CAPTION
        # ----------------------------------------------------

        if index <= len(captions):

            caption = captions[
                index - 1
            ]


            if caption:

                safe_caption = (

                    caption

                    .replace(
                        "&",
                        "&amp;"
                    )

                    .replace(
                        "<",
                        "&lt;"
                    )

                    .replace(
                        ">",
                        "&gt;"
                    )

                    .replace(
                        "\n",
                        "<br/>"
                    )

                )


                elements.append(

                    Paragraph(

                        safe_caption,

                        styles["Normal"]

                    )

                )


                elements.append(
                    Spacer(1, 8)
                )


        # ----------------------------------------------------
        # MODULE TEXT
        # ----------------------------------------------------

        elements.append(

            Paragraph(

                escape_pdf_text(module),

                styles["BodyText"]

            )

        )


        elements.append(
            Spacer(1, 15)
        )


    pdf.build(
        elements
    )


    return buffer.getvalue()


# ============================================================
# STORY VIDEO
# ============================================================

def create_story_video(
    images,
    module_audio,
    module_durations
):

    # --------------------------------------------------------
    # REQUIRE MATCHED DATA
    # --------------------------------------------------------

    if not images:

        raise ValueError(
            "No generated images available."
        )


    if not module_audio:

        raise ValueError(
            "No narration audio available."
        )


    if len(images) != len(module_audio):

        raise ValueError(
            "Number of images and narration modules do not match."
        )


    temp_dir = Path(

        tempfile.mkdtemp(
            prefix="ai_story_video_"
        )

    )


    silent_video = (
        temp_dir /
        "silent.mp4"
    )


    audio_file = (
        temp_dir /
        "narration.mp3"
    )


    final_video = (
        temp_dir /
        "AI_Story_Video.mp4"
    )


    width = 768

    height = 512

    fps = 20


    writer = imageio.get_writer(

        str(silent_video),

        fps=fps,

        codec="libx264",

        format="FFMPEG",

        pixelformat="yuv420p",

        macro_block_size=1

    )


    try:

        for index, image in enumerate(images):

            duration = (
                module_durations[index]
            )


            total_frames = max(

                int(
                    duration * fps
                ),

                int(
                    2 * fps
                )

            )


            base = (

                image

                .convert(
                    "RGB"
                )

                .resize(

                    (
                        width,
                        height
                    ),

                    Image.Resampling.LANCZOS

                )

            )


            for frame_number in range(
                total_frames
            ):

                progress = (

                    frame_number
                    /
                    max(
                        total_frames - 1,
                        1
                    )

                )


                # Gentle Ken Burns zoom
                zoom = (
                    1.0
                    +
                    0.04 * progress
                )


                crop_width = int(
                    width / zoom
                )


                crop_height = int(
                    height / zoom
                )


                left = (
                    width
                    -
                    crop_width
                ) // 2


                top = (
                    height
                    -
                    crop_height
                ) // 2


                frame = (

                    base

                    .crop(

                        (

                            left,

                            top,

                            left + crop_width,

                            top + crop_height

                        )

                    )

                    .resize(

                        (
                            width,
                            height
                        ),

                        Image.Resampling.LANCZOS

                    )

                )


                writer.append_data(
                    np.asarray(frame)
                )


    finally:

        writer.close()


    # --------------------------------------------------------
    # COMBINE NARRATION
    # --------------------------------------------------------

    complete_audio = combine_audio_files(
        module_audio
    )


    audio_file.write_bytes(
        complete_audio
    )


    # --------------------------------------------------------
    # MERGE VIDEO + AUDIO
    # --------------------------------------------------------

    ffmpeg = get_ffmpeg()


    command = [

        ffmpeg,

        "-y",

        "-i",
        str(silent_video),

        "-i",
        str(audio_file),

        "-map",
        "0:v:0",

        "-map",
        "1:a:0",

        "-c:v",
        "copy",

        "-c:a",
        "aac",

        "-b:a",
        "128k",

        "-shortest",

        str(final_video)

    ]


    result = subprocess.run(

        command,

        capture_output=True,

        text=True

    )


    if result.returncode != 0:

        raise RuntimeError(

            "FFmpeg video creation failed:\n"
            +
            result.stderr[-3000:]

        )


    return final_video.read_bytes()


# ============================================================
# RESET STORY
# ============================================================

def reset_story():

    for key, value in DEFAULT_STATE.items():

        st.session_state[key] = value


# ============================================================
# GENERATE STORY BUTTON
# ============================================================

if st.button(

    "✨ Generate Story",

    type="primary",

    use_container_width=True

):

    if not topic.strip():

        st.warning(
            "Please enter a story topic."
        )

        st.stop()


    # --------------------------------------------------------
    # CLEAR PREVIOUS STORY
    # --------------------------------------------------------

    reset_story()


    st.session_state.topic = topic

    st.session_state.style = style

    st.session_state.language = language


    # --------------------------------------------------------
    # STORY GENERATION
    # --------------------------------------------------------

    story_placeholder = st.empty()


    with story_placeholder.container():

        st.info(
            "📖 Creating your story..."
        )


    try:

        story = generate_story(
            topic
        )

    except Exception as e:

        story_placeholder.error(

            f"Story generation failed:\n\n{e}"

        )

        st.stop()


    modules = split_modules(
        story
    )


    # --------------------------------------------------------
    # SAVE IMMEDIATELY
    # --------------------------------------------------------

    st.session_state.story = story

    st.session_state.modules = modules

    st.session_state.story_generated = True


    # --------------------------------------------------------
    # DISPLAY STORY IMMEDIATELY
    # --------------------------------------------------------

    story_placeholder.empty()


    st.success(
        "📖 Story created successfully!"
    )


    st.subheader(
        "📖 Full Story"
    )


    st.write(
        story
    )


    st.subheader(
        f"🎨 Story Modules ({len(modules)})"
    )


    # --------------------------------------------------------
    # CHARACTER BIBLE
    # --------------------------------------------------------

    with st.spinner(
        "👤 Designing consistent characters..."
    ):

        try:

            character_bible = (
                create_character_bible(
                    story
                )
            )

        except Exception:

            character_bible = ""

            st.warning(
                "Character design could not be created. "
                "Images will still be generated."
            )


    st.session_state.character_bible = (
        character_bible
    )


    # --------------------------------------------------------
    # GENERATE EACH MODULE
    # --------------------------------------------------------

    for index, module in enumerate(
        modules,
        start=1
    ):

        st.markdown(
            f"## 📍 Module {index}/{len(modules)}"
        )


        # ----------------------------------------------------
        # SHOW MODULE CONTENT BEFORE IMAGE
        # ----------------------------------------------------

        st.info(
            module
        )


        image_placeholder = st.empty()


        image_placeholder.info(
            "🎨 Generating an illustration "
            "from this exact story moment..."
        )


        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        try:

            image = generate_image(

                module_text=module,

                character_bible=character_bible,

                style=style,

                story=story,

                module_index=index

            )


            st.session_state.generated_images.append(
                image
            )


            image_placeholder.image(

                image,

                caption=f"Module {index} illustration",

                use_container_width=True

            )


        except Exception as e:

            st.session_state.generated_images.append(
                None
            )


            image_placeholder.error(

                f"Image generation failed for "
                f"Module {index}:\n\n{e}"

            )


        # ----------------------------------------------------
        # CAPTION
        # ----------------------------------------------------

        try:

            caption = create_caption(
                module
            )

        except Exception:

            caption = ""


        st.session_state.generated_captions.append(
            caption
        )


        if caption:

            st.markdown(
                f"✨ {caption}"
            )


        st.markdown("---")


    # ========================================================
    # NARRATION
    # ========================================================

    st.subheader(
        "🔊 Full Narration"
    )


    try:

        with st.spinner(
            "🎙️ Generating narration for every module..."
        ):

            (
                module_audio,
                module_durations,
                full_audio,
                audio_source

            ) = generate_full_narration(

                modules,

                language

            )


        st.session_state.module_audio = (
            module_audio
        )


        st.session_state.module_durations = (
            module_durations
        )


        st.session_state.full_audio = (
            full_audio
        )


        st.session_state.audio_source = (
            audio_source
        )


        st.audio(
            full_audio,
            format="audio/mp3"
        )


        st.success(

            f"🎙️ Full narration generated "
            f"using {audio_source}."

        )


    except Exception as e:

        st.error(
            f"Narration generation failed:\n\n{e}"
        )


    # ========================================================
    # PDF
    # ========================================================

    try:

        with st.spinner(
            "📄 Creating illustrated PDF..."
        ):

            st.session_state.pdf_bytes = (
                create_pdf_document(

                    story,

                    topic,

                    modules,

                    st.session_state.generated_images,

                    st.session_state.generated_captions

                )
            )


        with st.spinner(
            "📝 Creating illustrated Word document..."
        ):

            st.session_state.word_bytes = (
                create_word_document(

                    story,

                    topic,

                    modules,

                    st.session_state.generated_images,

                    st.session_state.generated_captions

                )
            )


    except Exception as e:

        st.error(
            f"Document creation failed:\n\n{e}"
        )


    # ========================================================
    # VIDEO
    # ========================================================

    # Only create video if EVERY module has an image.
    # This prevents narration and images from becoming misaligned.

    all_images_available = (

        len(
            st.session_state.generated_images
        )
        ==
        len(modules)

        and

        all(
            image is not None
            for image in st.session_state.generated_images
        )

    )


    if (

        all_images_available

        and

        len(
            st.session_state.module_audio
        )
        ==
        len(modules)

    ):

        try:

            with st.spinner(
                "🎬 Creating full narrated story video..."
            ):

                st.session_state.video_bytes = (
                    create_story_video(

                        st.session_state.generated_images,

                        st.session_state.module_audio,

                        st.session_state.module_durations

                    )
                )


        except Exception as e:

            st.error(
                f"Video creation failed:\n\n{e}"
            )

    else:

        st.warning(

            "🎬 Video was not created because "
            "one or more story illustrations failed. "
            "Please generate the story again."

        )


    st.session_state.story_generated = True


    st.success(
        "🎉 Your complete illustrated storybook is ready!"
    )


# ============================================================
# DISPLAY SAVED STORY
# ============================================================

if st.session_state.story_generated:

    # ========================================================
    # FULL STORY
    # ========================================================

    st.subheader(
        "📖 Full Story"
    )


    st.write(
        st.session_state.story
    )


    # ========================================================
    # STORY MODULES
    # ========================================================

    st.subheader(

        f"🎨 Story Modules "
        f"({len(st.session_state.modules)})"

    )


    for index, module in enumerate(

        st.session_state.modules,

        start=1

    ):

        st.markdown(

            f"## 📍 Module {index}"

        )


        # ----------------------------------------------------
        # MODULE TEXT
        # ----------------------------------------------------

        st.info(
            module
        )


        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        if index <= len(
            st.session_state.generated_images
        ):

            image = (
                st.session_state.generated_images[
                    index - 1
                ]
            )


            if image is not None:

                st.image(

                    image,

                    caption=f"Module {index} illustration",

                    use_container_width=True

                )

            else:

                st.warning(
                    "Illustration unavailable for this module."
                )


        # ----------------------------------------------------
        # CAPTION
        # ----------------------------------------------------

        if index <= len(
            st.session_state.generated_captions
        ):

            caption = (
                st.session_state.generated_captions[
                    index - 1
                ]
            )


            if caption:

                st.markdown(
                    f"✨ {caption}"
                )


        # ----------------------------------------------------
        # DURATION
        # ----------------------------------------------------

        if index <= len(
            st.session_state.module_durations
        ):

            duration = (
                st.session_state.module_durations[
                    index - 1
                ]
            )


            st.caption(

                f"🎬 Narration duration: "
                f"{duration:.1f} seconds"

            )


        st.markdown("---")


    # ========================================================
    # NARRATION
    # ========================================================

    st.subheader(
        "🔊 Full Narration"
    )


    if st.session_state.full_audio:

        st.audio(

            st.session_state.full_audio,

            format="audio/mp3"

        )


        st.caption(

            "Narration engine: "
            +
            st.session_state.audio_source

        )


    # ========================================================
    # VIDEO
    # ========================================================

    st.subheader(
        "🎬 Full Story Video"
    )


    if st.session_state.video_bytes:

        st.video(
            st.session_state.video_bytes
        )


        total_duration = sum(

            st.session_state.module_durations

        )


        st.caption(

            f"Total video duration: "
            f"{total_duration:.1f} seconds"

        )


        st.info(

            "Each illustration remains on screen "
            "for the duration of its corresponding "
            "narration."

        )


    # ========================================================
    # DOWNLOAD MENU
    # ========================================================

    st.subheader(
        "📥 Storybook Downloads"
    )


    st.write(
        "Click below to choose what you want to download."
    )


    with st.popover(

        "📥 Download Storybook",

        use_container_width=True

    ):

        st.markdown(
            "### 📚 Download Options"
        )


        st.caption(
            "Choose any format."
        )


        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        if st.session_state.pdf_bytes:

            st.download_button(

                "📄 Illustrated PDF",

                data=st.session_state.pdf_bytes,

                file_name="AI_Story_Illustrated.pdf",

                mime="application/pdf",

                use_container_width=True,

                on_click="ignore"

            )


        # ----------------------------------------------------
        # WORD
        # ----------------------------------------------------

        if st.session_state.word_bytes:

            st.download_button(

                "📝 Illustrated Word",

                data=st.session_state.word_bytes,

                file_name="AI_Story_Illustrated.docx",

                mime=(
                    "application/vnd.openxmlformats-"
                    "officedocument.wordprocessingml.document"
                ),

                use_container_width=True,

                on_click="ignore"

            )


        # ----------------------------------------------------
        # AUDIO
        # ----------------------------------------------------

        if st.session_state.full_audio:

            st.download_button(

                "🔊 Full Narration MP3",

                data=st.session_state.full_audio,

                file_name="AI_Story_Narration.mp3",

                mime="audio/mpeg",

                use_container_width=True,

                on_click="ignore"

            )


        # ----------------------------------------------------
        # VIDEO
        # ----------------------------------------------------

        if st.session_state.video_bytes:

            st.download_button(

                "🎬 Full Story MP4",

                data=st.session_state.video_bytes,

                file_name="AI_Story_Video.mp4",

                mime="video/mp4",

                use_container_width=True,

                on_click="ignore"

            )