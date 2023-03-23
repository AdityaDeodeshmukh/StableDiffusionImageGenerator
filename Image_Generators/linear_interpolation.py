import argparse
import cv2
import os.path as path
import numpy as np
import torch
from PIL import Image
from Custom_Pipelines.pipeline_stable_diffusion_Mult_img2img import StableDiffusionMultImg2ImgPipeline
import imageio.v3 as iio
def revert_generator(generator,sd):
    generator=torch.manual_seed(0)
    if sd:
        generator=torch.manual_seed(int(sd))
    return generator


parser = argparse.ArgumentParser()
parser.add_argument("prompt",  help="Prompt")
parser.add_argument("image1",  help="Image1")
parser.add_argument("image2",  help="Image2")
parser.add_argument("-s","--seed",help="Seed for random number generator")
parser.add_argument("-d","--duration",help="Duration of each frame")
parser.add_argument("-n","--number",help="Number of frames")
parser.add_argument("-t","--type",help="Save format of the video")
parser.add_argument("-m","--model", help="Name of model")
args = parser.parse_args()
device = "cuda"
model_name="stable-diffusion-v1-5"
if args.model:
    model_name=args.model
pipe = StableDiffusionMultImg2ImgPipeline.from_pretrained("Models/"+model_name, torch_dtype=torch.float16)
pipe = pipe.to(device)
init_image = Image.open("Stored_Images/"+args.image1)
init_image = init_image.resize((768,512))
init_image1 = Image.open("Stored_Images/"+args.image2)
init_image1 = init_image1.resize((768,512))
prompt = args.prompt
generator=torch.manual_seed(0)
generator=revert_generator(generator,args.seed)


images = pipe(prompt=prompt, image1=init_image, image2=init_image1, 
                strength=0.75, guidance_scale=7.5, weight1=1, 
                weight2=0,generator=generator,safety = False).images
images[0].show()
generator=revert_generator(generator,args.seed)
images = pipe(prompt=prompt, image1=init_image, image2=init_image1, 
                strength=0.75, guidance_scale=7.5, weight1=0, 
                weight2=1,generator=generator,safety = False).images
images[0].show()

x = input("Do you want to continue? (y/n)")
if x=="n":
    exit()

images1=[]
n=100
if args.number:
    n=int(args.number)
duration = 5
if args.duration:
    duration = int(args.duration)


for i in range(n+1):
    w1 = i/n
    w2 = 1-w1
    generator=revert_generator(generator,args.seed)
    images = pipe(prompt=prompt, image1=init_image, image2=init_image1, 
                strength=0.75, guidance_scale=7.5, weight1=w1 , 
                weight2=w2,generator=generator,safety = False).images
    if i%10==0:
        print("===================| Iteration: ",i,"|===================")
    images1.append(np.array(images[0]))
print("Generation Done")

ext ="gif"
if args.type == "mp4":
    ext = "mp4"

if ext == "gif":
    i=0
    while(path.exists("./movies/movie"+str(i)+".gif")):
        i+=1
    iio.imwrite("./movies/movie"+str(i)+".gif", images1,duration = duration)
else:
    i=0
    while(path.exists("./movies/movie"+str(i)+".avi")):
        i+=1
    pth="./movies/movie"+str(i)+".avi"
    fps = 1000/duration
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter("./movies/movie"+str(i)+".avi", fourcc, float(fps), (768, 512))
    for image in images1:
        img=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(img)
    for i in range(50):
        video.write(img)