import openai
import vertexai
from vertexai.vision_models import ImageTextModel, Image
from google.cloud import vision
import os
import tempfile
import streamlit as st

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ID = st.secrets['Project_id']
LOCATION = 'us-central1'
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = ImageTextModel.from_pretrained("imagetext@001")
st.title("Image Alt Text and Description Generator")

def detect_safe_search(image_path):
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.safe_search_detection(image=image)
    safe = response.safe_search_annotation
    likelihood_name = (
        "UNKNOWN",
        "VERY_UNLIKELY",
        "UNLIKELY",
        "POSSIBLE",
        "LIKELY",
        "VERY_LIKELY",
    )

    return likelihood_name[safe.adult], likelihood_name[safe.violence]


def generateDesc(para):
    feeder = "Give me a elaborate description of the facts for this image's alt text: " + para + " use all the above information but do not add any analysis or opinions that isn't a part of the alt text"
    formatter = "just keep it simple without any poetic essence but make it about 50 words. write in a way that a person who doesn't understand emotions is able to imagine the contents of the image"
    instruction = "start with 'A picture of '"
    conversation = [
        {"role": "system", "content": "You are an AI language model."},
        {"role": "user", "content": feeder},
        {"role": "user", "content": formatter},
        {"role": "user", "content": instruction},
    ]
    openai.api_key = st.secrets['openai_key']
    response = openai.ChatCompletion.create(
        model="gpt-4", messages=conversation,)
    return response.choices[0].message.content


def predict(image):
    adult, violence = detect_safe_search(image)
    if adult == 'VERY_LIKELY' or violence == 'VERY_LIKELY':
      txt = 'uploaded image is likely to contain either adult or violence content'
      return txt

    source_image = Image.load_from_file(location=image)
    captions = model.get_captions(
        image=source_image,
        number_of_results=1,
        language="en",
    )

    ans1 = model.ask_question(
        image=source_image,
        question="Give me all the information about the people in this image.",
        number_of_results=1,
    )
    ans2 = model.ask_question(
        image=source_image,
        question="What does this image contain?",
        number_of_results=1,
    )
    ans3 = model.ask_question(
        image=source_image,
        question="Give me all the text in this image",
        number_of_results=1,
    )

    Answers = ans1 + ans2 + ans3
    Answers = list(set(Answers))
    ans = ""
    for x in Answers:
        if x == "unanswered" or x == "no text in image" or x == "no":
            pass
        else:
            ans = ans + ' ' + x
    return captions[0].capitalize(), generateDesc(captions[0]+' '+ans)

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "jpeg"])
if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image")
    if st.button("Generate Alt Text and Description"):
        with tempfile.NamedTemporaryFile(delete=False) as temp_image:
            temp_image.write(uploaded_image.read())
            temp_image_path = temp_image.name
        alt_text, description = predict(temp_image_path)
        st.write(f"Alt Text: {alt_text}")
        st.write(f"Description: {description}")

st.text("Please note that this tool uses AI models for content analysis and generation. The accuracy may vary.")
