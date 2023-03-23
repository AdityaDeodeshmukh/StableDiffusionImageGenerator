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
from Custom_Pipelines.pipeline_stable_diffusion_Mult_img2img import StableDiffusionMultImg2ImgPipeline

def image_from_mult_image_pipeline(model_name,opt=["o"]):
    pipe = StableDiffusionMultImg2ImgPipeline.from_pretrained("Models/"+model_name, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe = tune(opt,pipe)
    return pipe

def image_from_mult_images(prompt,image1,image2,pipe,strength=0.75,guidance_scale=7.5,
                   num_inferences=50,num_images=1,height=512,width=768,seed=0,weight1=0.5,weight2=0.5,safety=False):
    image1=image1.resize((width,height))
    image2=image2.resize((width,height))
    if seed:
        generator = torch.manual_seed(seed)
        images = pipe(prompt,num_images_per_prompt=num_images,
                  num_inference_steps=num_inferences, generator=generator,
                  weight1=weight1,weight2=weight2, safety=safety,
                  image1=image1,image2=image2, strength=strength,guidance_scale=guidance_scale)
        return images
    generator = torch.manual_seed(0)
    images = pipe(prompt,num_images_per_prompt=num_images,
                  num_inference_steps=num_inferences,
                  weight1=weight1,weight2=weight2, generator=generator,safety=safety,
                  image1=image1,image2=image2,strength=strength,guidance_scale=guidance_scale)
    return images

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    torch.cuda.empty_cache()
    parser.add_argument("prompt",  help="Prompt")
    parser.add_argument("image1",  help="First Image")
    parser.add_argument("image2",  help="Second Image")
    parser.add_argument("-w1", "--weight1", help="Weight of first image")
    parser.add_argument("-w2", "--weight2", help="Weight of second image")
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
    WIDTH=768
    NAME="img1"
    MODEL="stable-diffusion-v1-5"
    NUM_INFERENCES=50
    NUM_IMAGES=1
    IMAGE1=Image.open("Stored_Images/"+args.image1)
    IMAGE2=Image.open("Stored_Images/"+args.image2)
    GUIDANCE_SCALE=7.5
    STRENGTH=0.75
    if not args.weight1:
        args.weight1=0.5
    if not args.weight2:
        args.weight2=0.5
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
   
    pipeline = image_from_mult_image_pipeline(model_name=MODEL,opt=optimizers)
    if args.seed:
        generated_images = image_from_mult_images(prompt=args.prompt,pipe=pipeline,
                         num_inferences=NUM_INFERENCES,image1=IMAGE1,image2=IMAGE2,
                         num_images=NUM_IMAGES,height=HEIGHT,width=WIDTH,
                         strength=STRENGTH,guidance_scale=GUIDANCE_SCALE,
                         seed=int(args.seed))
    else:
        generated_images = image_from_mult_images(prompt=args.prompt,pipe=pipeline,
                            num_inferences=NUM_INFERENCES,image1=IMAGE1,image2=IMAGE2,
                            weight1=float(args.weight1),weight2=float(args.weight2),
                            num_images=NUM_IMAGES,height=HEIGHT,width=WIDTH,
                            strength=STRENGTH,guidance_scale=GUIDANCE_SCALE)
    generated_images=generated_images[0]
    for n,img in enumerate(generated_images):
        img.save("./Generated_Images/"+NAME+"("+str(n)+")"+".png")
    if args.display:
        for img in generated_images:
            img.show()