from groq import Groq
import base64
import mimetypes
import os

class GroqVisionWrapper:
    """
    Wrapper custom untuk model multimodal (Vision) Groq
    agar bisa dipakai di QueryProcessor layaknya LLM biasa.
    """

    def __init__(self, api_key: str, model: str = "llama-3.2-11b-vision-preview"):
        self.client = Groq(api_key=api_key)
        self.model = model

    def complete(self, prompt: str, image_url: str | None = None, local_image_path: str | None = None):
        """Kirim prompt + (opsional) image_url ATAU local_image_path ke Groq Vision model."""
        content = [{"type": "text", "text": prompt}]
        
        if image_url:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        elif local_image_path and os.path.exists(local_image_path):
            try:
                mime_type, _ = mimetypes.guess_type(local_image_path)
                if not mime_type or not mime_type.startswith("image/"):
                    file_ext = os.path.splitext(local_image_path)[1].lower()
                    if file_ext == ".png":
                        mime_type = "image/png"
                    elif file_ext == ".webp":
                        mime_type = "image/webp"
                    else:
                        mime_type = "image/jpeg"

                with open(local_image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

                image_url_data = f"data:{mime_type};base64,{base64_image}"
                
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url_data}
                })
                print(f"Successfully encoded local image {local_image_path} for vision analysis.")
            except Exception as e:
                print(f"Error encoding local image to base64: {e}")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
        )

        return response.choices[0].message.content.strip()