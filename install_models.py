"""A python script to install image models from the huggingface hub
    Options: 
    CompVis/stable-diffusion-v1-4
    runwayml/stable-diffusion-v1-5
    stabilityai/stable-diffusion-2-base
    stabilityai/stable-diffusion-2
    stabilityai/stable-diffusion-2-1-base
    stabilityai/stable-diffusion-2-1

    To Run:
    Store your huggingface auth token in a file named auth_token.txt
    python install_models -mdl <Option>
"""
import argparse
from diffusers import DiffusionPipeline

if __name__=='__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-mdl", "--model", help="AI generator model")
    args = argParser.parse_args()
    repo_id=args.model
    with open('auth_token.txt',encoding="utf-8") as f:
        Auth_token = f.readlines()
    Auth_token=Auth_token[0]
    mod=repo_id.split("/")[-1]
    print(mod)
    stable_diffusion = DiffusionPipeline.from_pretrained(repo_id, use_auth_token=Auth_token)
    DiffusionPipeline.save_pretrained(stable_diffusion,"Models/"+mod)
