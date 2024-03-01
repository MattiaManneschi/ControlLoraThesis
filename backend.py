from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionUpscalePipeline, \
    DDIMScheduler
import torch
from controlnet_aux import HEDdetector
import streamlit as st
from matplotlib import pyplot as plt

torch.cuda.empty_cache()


def createPipe(controlnet):
    pipe = StableDiffusionControlNetPipeline.from_single_file("/home/mattia/Desktop/models/arthemyObjects_v10"
                                                              ".safetensors", controlnet=controlnet,
                                                              safety_checker=None, torch_dtype=torch.float16).to("cuda")

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_model_cpu_offload()

    pipe.load_lora_weights("/home/mattia/Desktop/models/more_details.safetensors")
    pipe.load_textual_inversion("/home/mattia/Desktop/models/BadDream.pt")
    pipe.load_textual_inversion("/home/mattia/Desktop/models/verybadimagenegative_v1.3.pt")

    return pipe


def scribbleInf(image, p, neg_p, n):
    # processor = HEDdetector.from_pretrained('lllyasviel/Annotators')

    # plt.imshow(image)
    # plt.show()

    # control_image = processor(image, scribble=True)

    # plt.imshow(control_image)
    # plt.show()

    image = Image.fromarray(image, 'RGB')

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_scribble", torch_dtype=torch.float16)

    pipe = createPipe(controlnet)

    generator = torch.manual_seed(0)

    images = []

    for i in range(n):
        st.write("processing image {}....".format(i+1))
        image = pipe(
            prompt=p + "product shot, finely detailed, purism, art, 100mm advertising photography, studio lighting, "
                       "8k, hyperdetailed",
            negative_prompt=neg_p + "BadDream, (fake), high contrast, oversharp, repetition, boring, emotionless, "
                                    "verybadimagenegative_v1.3",
            num_inference_steps=35,
            generator=generator,
            image=image,
            guidance_scale=7.5
        ).images[0]
        images.append(image)

    return images


def upscale(image):
    upscaling_model = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(upscaling_model, torch_dtype=torch.float32).to("cuda")

    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.enable_attention_slicing()

    pipeline.enable_model_cpu_offload()

    prompt = "UHD, 4k, hyper realistic, extremely detailed, professional, vibrant, not grainy, smooth"

    upscaled_image = pipeline(prompt=prompt, image=image).images[0]
