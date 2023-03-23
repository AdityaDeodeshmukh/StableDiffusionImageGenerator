"""Image Generation Module
    Used to generate images using the Stable Diffusion Pipeline
    To use:
    pipeline = image_to_image_pipeline(<Model_Name>,[options]) 
    images = pipe(prompt,pipe)
    
    To call from terminal:
    Generate_Modules/image_to_image.py <prompt> <image>"""
import torch
import argparse
from tune import tune
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
def image_to_image_pipeline(model_name,opt=["o"]):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained("Models/"+model_name, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe = tune(opt,pipe)
    return pipe

def image_to_images(prompt,image,pipe,strength=0.75,guidance_scale=7.5,
                   num_inferences=50,num_images=1,height=512,width=512,seed=0):
    image=image.resize((width,height))
    if seed:
        generator = torch.manual_seed(seed)
        images = pipe(prompt,num_images_per_prompt=num_images,
                  num_inference_steps=num_inferences, generator=generator,
                  image=image, strength=strength,guidance_scale=guidance_scale)
        return images
    image=image.resize((width,height))
    images = pipe(prompt,num_images_per_prompt=num_images,
                  num_inference_steps=num_inferences,
                  image=image, strength=strength,guidance_scale=guidance_scale)
    return images

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    torch.cuda.empty_cache()
    parser.add_argument("prompt",  help="Prompt")
    parser.add_argument("image",  help="Image")
    parser.add_argument("-hg", "--height", help="Height of output image")
    parser.add_argument("-wd", "--width", help="Width of output image")
    parser.add_argument("-nm","--name", help="Name of image")
    parser.add_argument("-m","--model", help="Name of model")
    parser.add_argument("-i","--inferences", help="The number of step inferences")
    parser.add_argument("-num","--number",help="Number of images per prompt")
    parser.add_argument("-s","--seed",help="Seed for random number generator")
    parser.add_argument("-str","--strength",help="Strength of the prompt")
    parser.add_argument("-g","--guidance",help="Guidance scale of the prompt")
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
    IMAGE=Image.open("Stored_Images/"+args.image)
    GUIDANCE_SCALE=7.5
    STRENGTH=0.75
    if args.height:
        HEIGHT=int(args.height)
    if args.width:
        WIDTH=int(args.width)
    if args.name:
        NAME=args.name
    if args.inferences:
        NUM_INFERENCES=int(args.inferences)
    if args.number:
        NUM_IMAGES=int(args.number)
    optimizers=[]
    if args.vaeslicing:
        optimizers.append('v')
    if args.offload:
        optimizers.append('o')
    if args.attention:
        optimizers.append('a')
    if args.strength:
        STRENGTH=float(args.strength)
    if args.guidance:
        GUIDANCE_SCALE=float(args.guidance)
   
    pipeline = image_to_image_pipeline(model_name=MODEL,opt=optimizers)
    if args.seed:
        generated_images = image_to_images(prompt=args.prompt,pipe=pipeline,
                         num_inferences=NUM_INFERENCES,image=IMAGE,
                         num_images=NUM_IMAGES,height=HEIGHT,width=WIDTH,
                         strength=STRENGTH,guidance_scale=GUIDANCE_SCALE,
                         seed=int(args.seed))
    else:
        generated_images = image_to_images(prompt=args.prompt,pipe=pipeline,
                            num_inferences=NUM_INFERENCES,image=IMAGE,
                            num_images=NUM_IMAGES,height=HEIGHT,width=WIDTH,
                            strength=STRENGTH,guidance_scale=GUIDANCE_SCALE)
    generated_images=generated_images[0]
    for n,img in enumerate(generated_images):
        img.save("./Generated_Images/"+NAME+"("+str(n)+")"+".png")
    if args.display:
        for img in generated_images:
            img.show()