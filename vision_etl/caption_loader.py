import os
import base64
from io import BytesIO
from PIL import Image
import torch
from open_clip import create_model_and_transforms, get_tokenizer
from openai import AzureOpenAI
from dotenv import load_dotenv


class CaptionLoader:
    """
    Unified CaptionLoader:
    - CLIP text embedding mode (`mode='clip'`)
    - GPT-4o image caption generation (`mode='azure'`)
    """

    def __init__(self, model_name: str = "ViT-L-14", device: str = "cuda", mode: str = "clip"):
        self.mode = mode.lower()
        self.device = device if torch.cuda.is_available() else "cpu"

        if self.mode == "clip":
            self.model_name = model_name
            self.model, _, _ = create_model_and_transforms(model_name, pretrained="laion2b_s32b_b82k")
            self.model = self.model.to(self.device).eval()
            self.tokenizer = get_tokenizer(model_name)

        elif self.mode == "azure":
            load_dotenv()
            self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
            self.api_base = os.getenv("AZURE_OPENAI_API_BASE")
            self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
            self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

            if not all([self.api_key, self.api_base, self.api_version]):
                raise EnvironmentError("Missing required Azure OpenAI environment variables. Please check your .env file.")

            self.client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.api_base,
                api_version=self.api_version,
            )

        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Use 'clip' or 'azure'.")

    def get_caption_embeddings(self, captions: list[str]) -> torch.Tensor:
        """
        For mode='clip': Generate normalized CLIP text embeddings.
        """
        if self.mode != "clip":
            raise RuntimeError("get_caption_embeddings is only available in 'clip' mode.")

        tokens = self.tokenizer(captions).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb  # Shape: (N, D)

    def encode_image_to_base64(self, img: Image.Image) -> str:
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def generate_caption(self, img: Image.Image) -> str:
        """
        For mode='azure': Generate natural language caption from retinal image.
        """
        if self.mode != "azure":
            raise RuntimeError("generate_caption is only available in 'azure' mode.")

        img_base64 = self.encode_image_to_base64(img)
        prompt = "Describe this retinal image briefly for a diagnosis model. Be precise and use medical vocabulary."

        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a medical expert trained to describe retina fundus images."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    ]
                }
            ],
            temperature=0,
            max_tokens=80
        )

        return response.choices[0].message.content.strip()


# --- TESTING ---
if __name__ == "__main__":
    # CLIP test
    clip_loader = CaptionLoader(mode="clip")
    captions = [
        "No diabetic retinopathy",
        "Severe non-proliferative diabetic retinopathy",
        "Proliferative DR"
    ]
    emb = clip_loader.get_caption_embeddings(captions)
    print("CLIP Embedding shape:", emb.shape)