import csv
import io
import time
import random
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
import base64
import json
from tortoise.contrib.fastapi import register_tortoise
import openai
import difflib
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.ocr.v20181119 import ocr_client, models

from fastapi.middleware.cors import CORSMiddleware

from functools import wraps
from typing import Callable

from solve_gpt_api.models import Question

ROOT_PATH = "/solvegpt/api/v1"


def log_endpoint_data(func: Callable):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        print(f"Params: {kwargs}")

        response = await func(*args, **kwargs)

        response_data = {
            "status_code": response.status_code,
        }
        try:
            response_content = response.body
            response_data["body"] = json.loads(response_content)
        except (json.JSONDecodeError, AttributeError):
            pass

        print("Response:", response_data)

        return response

    return wrapper


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("/root/solvegpt/openai_config.json", encoding="utf-8") as f:
    openai_config = json.loads(f.read())


class ResponseBean:
    def __init__(self, success, message: str, data):
        self.success = success
        self.message = message
        self.data = data

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def success(msg="", data=None) -> dict:
        return ResponseBean(True, msg, data).to_dict()

    @staticmethod
    def fail(msg, data=None) -> dict:
        return ResponseBean(False, msg, data).to_dict()


def generate_accuracy():
    accuracy_range = None
    rand_num = random.uniform(0, 1)

    if rand_num <= 0.9:
        accuracy_range = (90, 99)
    elif 0.9 < rand_num <= 0.98:
        accuracy_range = (80, 89)
    else:
        accuracy_range = (0, 79)

    return random.randint(accuracy_range[0], accuracy_range[1])


def tencent_ocr(img_base64):
    cred = credential.Credential(
        "", ""
    )
    httpProfile = HttpProfile()
    httpProfile.endpoint = "ocr.tencentcloudapi.com"
    clientProfile = ClientProfile()
    clientProfile.httpProfile = httpProfile
    client = ocr_client.OcrClient(cred, "", clientProfile)
    client = ocr_client.OcrClient(cred, "ap-shanghai", clientProfile)
    req = models.GeneralAccurateOCRRequest()
    params = {"ImageBase64": img_base64}
    req.from_json_string(json.dumps(params))
    resp = client.GeneralAccurateOCR(req)
    return resp.TextDetections


def get_openai_answer(question, model):
    openai.api_key = openai_config["api_key"]
    openai.api_base = openai_config["base_url"]

    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": openai_config["question_system_role"],
            },
            {"role": "user", "content": question},
        ],
        temperature=0.5,
    )
    first_answer = completion.choices[0].message.content

    # print(first_answer)

    # check_content = openai_config["check_prompt"].format(question, first_answer)
    # print(check_content)

    # completion = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": check_content
    #         }
    #     ],
    # )

    # final_answer = completion.choices[0].message.content
    return first_answer


async def get_answer_internal(question, model) -> Question:
    question_records = await Question.all()
    questions = [record.question for record in question_records]

    if questions.__len__() != 0:
        # vectorizer = TfidfVectorizer()
        # tfidf_matrix = vectorizer.fit_transform(questions)
        # target = vectorizer.transform([question])

        # cosine_similarities = cosine_similarity(target, tfidf_matrix)
        # max_index = cosine_similarities.argmax()

        # similarity_threshold = 0.9

        # if cosine_similarities[0, max_index] > similarity_threshold:
        #     return question_records[max_index].answer

        diff_ratios = [
            difflib.SequenceMatcher(
                lambda x: x in r"""!"#$%&'(),.:;?@[\]^_`{|}~""" or x.isspace(),
                question,
                q,
            ).ratio()
            for q in questions
        ]
        max_index = diff_ratios.index(max(diff_ratios))
        similarity_threshold = 0.95
        if diff_ratios[max_index] > similarity_threshold:
            return Question(
                question=question,
                answer=question_records[max_index].answer,
                accuracy=100,
            )

    return Question(
        question=question,
        answer=get_openai_answer(question, model),
        accuracy=generate_accuracy(),
    )


@app.get("/hello")
async def hello():
    return {"message": "Hello World"}


@app.post(f"{ROOT_PATH}/submitImage")
@log_endpoint_data
async def submit_image(image: UploadFile = File(...)):
    contents = await image.read()
    img_base64 = base64.b64encode(contents).decode("utf-8")

    try:
        text_detections = tencent_ocr(img_base64)
    except Exception as err:
        return JSONResponse(ResponseBean.fail(str(err)))

    if text_detections is None:
        return JSONResponse(ResponseBean.fail("No text detected."))

    detected_text = " ".join([item.DetectedText for item in text_detections])

    return_question = Question(question=detected_text).to_json()

    return JSONResponse(ResponseBean.success("Text detected.", return_question))
    # answer = await get_answer_internal(detected_text)
    # q = Question(question=detected_text, answer=answer)
    # return JSONResponse(content=ResponseBean(True, "Answer generated.", q.to_json()).to_dict())


@app.post(f"{ROOT_PATH}/submitText")
@log_endpoint_data
async def submit_text(text: str = Form(...), model: str = Form(...)):
    try:
        question = await get_answer_internal(text, model)
        message = "Answer generated from database." if question.accuracy == 100 else "Answer generated from SolveGPT."
        return JSONResponse(ResponseBean.success(message, question.to_json()))
    except Exception as e:
        return JSONResponse(ResponseBean.fail(str(e)))


@app.post(f"{ROOT_PATH}/save")
async def save_question(question: str = Form(...), answer: str = Form(...)):
    q = Question(question=question, answer=answer)
    await q.save()
    return JSONResponse(ResponseBean.success("Question saved."))


@app.get(f"{ROOT_PATH}/getAll")
async def get_all():
    question_records = await Question.all()
    questions = [record.to_json() for record in question_records]
    return JSONResponse(ResponseBean.success("Questions retrieved.", questions))


@app.post(f"{ROOT_PATH}/clear")
async def clear():
    await Question.all().delete()
    return JSONResponse(ResponseBean.success("Questions cleared.", None))


@app.get(f"{ROOT_PATH}/download")
async def download():
    questions = await Question.all().values("question", "answer")
    output = io.StringIO()
    csv_writer = csv.writer(output, delimiter="\t")
    for question in questions:
        csv_writer.writerow([question["question"], question["answer"]])
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="questions_{time.time()}.csv"'
        },
    )


@app.post(f"{ROOT_PATH}/upload")
async def upload(file: UploadFile = File(...)):
    file_content = await file.read()
    content = io.StringIO(file_content.decode())
    csv_reader = csv.reader(content, delimiter="\t")

    try:
        questions = [Question(question=row[0], answer=row[1]) for row in csv_reader]
    except IndexError:
        return JSONResponse(ResponseBean.fail("File format incorrect."))

    await Question.all().delete()
    await Question.bulk_create(questions)

    return ResponseBean.success("Questions uploaded.")


TORTOISE_ORM = {
    "connections": {"default": "sqlite://db.sqlite3"},
    "apps": {
        "models": {
            "models": ["solve_gpt_api.models"],
            "default_connection": "default",
        },
    },
}

register_tortoise(
    app,
    generate_schemas=True,
    add_exception_handlers=True,
    config=TORTOISE_ORM,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
