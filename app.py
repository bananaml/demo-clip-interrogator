from potassium import Potassium, Request, Response
import base64
from PIL import Image
from io import BytesIO
from clip_interrogator import Config, Interrogator

app = Potassium("clip-interrogator")

@app.init
def init():
    ci = Interrogator(
        Config(   
            clip_model_name="ViT-bigG-14/laion2b_s39b_b160k",
        )
    )
   
    context = {
        "ci": ci
    }
    return context

@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    ci = context.get("ci")
    image = request.json.get("image")
    img = image.encode('utf-8')
    img = BytesIO(base64.b64decode(img))
    img = Image.open(img)
    clip_txt = ci.interrogate(img)
    return Response(
        json = {"outputs": clip_txt}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()