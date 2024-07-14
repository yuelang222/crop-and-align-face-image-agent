import os
import os.path as osp
import argparse
from tqdm import tqdm
import torch
import numpy as np
import cv2
import PIL.Image
import PIL.ImageFile
from PIL import Image
import scipy.ndimage
from lib.landmarks_pytorch import LandmarksEstimation

import logging
logging.getLogger().setLevel(logging.ERROR)

#------------------tool---------------
IMAGE_EXT = ('.jpg', '.jpeg', '.png')

from langchain.pydantic_v1 import BaseModel, Field
class Input(BaseModel):
    input_dir: str = Field(description = 'image path')

from langchain.tools import BaseTool
from typing import Type
class Image_Align(BaseTool):
    name = 'image_crop_and_align'
    description = "use this tool when you need to align image,the input is image path,the output is aligned image path"
    args_schema: Type[BaseModel] = Input

    def align_crop_image(self, image, landmarks, transform_size=256):
        # Get estimated landmarks
        lm = landmarks
        lm_chin = lm[0: 17]            # left-right
        lm_eyebrow_left = lm[17: 22]   # left-right
        lm_eyebrow_right = lm[22: 27]  # left-right
        lm_nose = lm[27: 31]           # top-down
        lm_nostrils = lm[31: 36]       # top-down
        lm_eye_left = lm[36: 42]       # left-clockwise
        lm_eye_right = lm[42: 48]      # left-clockwise
        lm_mouth_outer = lm[48: 60]    # left-clockwise
        lm_mouth_inner = lm[60: 68]    # left-clockwise

        # Calculate auxiliary vectors
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        img = Image.fromarray(image)
        shrink = int(np.floor(qsize / transform_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, Image.Resampling.LANCZOS)
            quad /= shrink
            qsize /= shrink

        # Crop
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
            max(pad[3] - img.size[1] + border, 0))
        enable_padding = True
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            # mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
            #                   1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / (pad[0] + 1e-12), np.float32(w - 1 - x) / (pad[2] + 1e-12)),
                            1.0 - np.minimum(np.float32(y) / (pad[1] + 1e-12), np.float32(h - 1 - y) / (pad[3] + 1e-12)))

            blur = qsize * 0.01
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')

            quad += pad[:2]

        # Transform
        img = img.transform((transform_size, transform_size), Image.Transform.QUAD, (quad + 0.5).flatten(),
                            Image.Resampling.BILINEAR)

        return np.array(img)

    def read_image_opencv(self, image_path):
        # Read image in BGR order
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('uint8')

    def _run(self, input_dir):
        transform_size = 256

        # Get input/output directories
        input_dir = osp.abspath(osp.expanduser(input_dir))
        output_dir = osp.join(osp.split(input_dir)[0], "{}_aligned".format(osp.split(input_dir)[1]))

        print(f"#. 创建输出目录: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Get input images paths
        input_images = [osp.join(input_dir, dI) for dI in os.listdir(input_dir)
                        if osp.isfile(osp.join(input_dir, dI)) and osp.splitext(dI)[-1] in IMAGE_EXT]
        input_images.sort()

        # Build landmark estimator
        le = LandmarksEstimation(type='2D')

        for img_file in tqdm(input_images, desc='Preprocess {} images'.format(len(input_images))):
            # Open input image
            img = self.read_image_opencv(img_file).copy()

            # Landmark estimation
            img_tensor = torch.tensor(np.transpose(img, (2, 0, 1))).float().cuda()
            with torch.no_grad():
                landmarks, detected_faces = le.detect_landmarks(img_tensor.unsqueeze(0),detected_faces=None)
            # Align and crop face
            if len(landmarks) > 0:
                img = self.align_crop_image(image=img,
                                    landmarks=np.asarray(landmarks[0].detach().cpu().numpy()),
                                    transform_size=transform_size)
            else:
                print("#. Warning: No landmarks found in {}".format(img_file))
                with open('issues.txt', 'a' if osp.exists('issues.txt') else 'w') as f:
                    f.write("{}\n".format(img_file))

            # Save output image
            cv2.imwrite(osp.join(output_dir, osp.split(img_file)[-1]), cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR))

        return 'FFHQFaceAlignment/input_images_aligned'

    def _arun(self, input_dir):
        raise NotImplementedError("This tool does not support async")

tools = [Image_Align()]



#---------------llm
from langchain_openai import ChatOpenAI
import os
os.environ['OPENAI_API_KEY']="sk-oHD3AclmGGW3T1yl03E396Da555d4a7e82279f7c92070194"
os.environ['OPENAI_API_BASE'] = "https://api.xiaoai.plus/v1"

llm = ChatOpenAI(model='gpt-3.5-turbo-0125')



#---------------prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ('system', 'you are a helpful assistant,if user ask you question about math,you must use tool'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{user_input}'),
    MessagesPlaceholder(variable_name='agent_scratchpad')
])



#---------------agent
from langchain.agents import create_tool_calling_agent

agent = create_tool_calling_agent(llm, tools, prompt)

from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)



#----------------memory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage

store = {}

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="user_input",
    history_messages_key="chat_history",
)

session_id = '1'

def run():
    while True:
        user_input = input('user:')
        if user_input.lower() in ['exit','quit', 'over']:
            break

        response = agent_with_chat_history.invoke(
            {'user_input': user_input},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"ai  : {response['output']}")


if __name__=='__main__':
    run()