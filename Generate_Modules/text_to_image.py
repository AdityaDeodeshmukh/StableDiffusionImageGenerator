"""Image Generation Module
    Used to generate images using the Stable Diffusion Pipeline
    To use:
    pipeline = text_to_image_pipeline(<Model_Name>,[options]) 
    images = pipe(prompt,pipe)
    
    To call from terminal:
    Generate_Modules/text_to_image.py <prompt>"""
import argparse
import torch
from diffusers import StableDiffusionPipeline
#tuning the pipe to work on lower end hardware
def tune(opt,pipe):
    torch.backends.cudnn.benchmark = True
    for i in opt:
        if i=="v":
            pipe.enable_vae_slicing()
            continue
        if i=="o":
            pipe.enable_sequential_cpu_offload()
            continue
        if i=="a":
            pipe.enable_attention_slicing(1)
            continue
    return pipe
#Used to generate a pipeline from the given model
def text_to_image_pipeline(model_name,opt=["o"]):
    pipe = StableDiffusionPipeline.from_pretrained("Models/"+model_name, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe = tune(opt,pipe)
    return pipe
#Used to generate images using pipe created from above function
def text_to_images(prompt,pipe,num_inferences=50,num_images=1,height=512,width=512):
    images = pipe(prompt,num_images_per_prompt=num_images,
                  num_inference_steps=num_inferences,height=height,width=width)
    return images

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    torch.cuda.empty_cache()
    parser.add_argument("prompt",  help="Prompt")
    parser.add_argument("-hg", "--height", help="Height of output image")
    parser.add_argument("-wd", "--width", help="Width of output image")
    parser.add_argument("-nm","--name", help="Name of image")
    parser.add_argument("-m","--model", help="Name of model")
    parser.add_argument("-i","--inferences", help="The number of step inferences")
    parser.add_argument("-num","--number",help="Number of images per prompt")
    parser.add_argument("-d","--display",
                        help="To display the image",action="store_true")
    parser.add_argument("-v","--vaeslicing",
                        help="To enable vae slicing (Reduces memory usage)",
                        action="store_true")
    parser.add_argument("-o","--offload",
                        help="To enable cpu offloading (Reduces memory usage)",
                        action="store_true")
    parser.add_argument("-a","--attention",
                        help="To enable attention slicing (Reduces memory usage)",
                        action="store_true")
    args = parser.parse_args()
    HEIGHT=512
    WIDTH=512
    NAME="img1"
    MODEL="stable-diffusion-v1-5"
    NUM_INFERENCES=50
    NUM_IMAGES=1
    if args.height:
        HEIGHT=int(args.height)
    if args.width:
        WIDTH=int(args.width)
    if args.name:
        NAME=args.name
    if args.inferences:
        NUN_INFERENCES=int(args.inferences)
    if args.number:
        NUM_IMAGES=int(args.number)
    optimizers=[]
    if args.vaeslicing:
        optimizers.append('v')
    if args.offload:
        optimizers.append('o')
    if args.attention:
        optimizers.append('a')
    pipeline = text_to_image_pipeline(model_name=MODEL,opt=optimizers)
    generated_images = text_to_images(prompt=args.prompt,pipe=pipeline,
                         num_inferences=NUM_INFERENCES,
                         num_images=NUM_IMAGES,height=HEIGHT,width=WIDTH)
    generated_images=generated_images[0]
    for n,img in enumerate(generated_images):
        img.save("./Generated_Images/"+NAME+"("+str(n)+")"+".png")
    if args.display:
        for img in generated_images:
            img.show()
