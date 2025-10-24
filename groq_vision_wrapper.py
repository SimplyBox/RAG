from groq import Groq

class GroqVisionWrapper:
    """
    Wrapper custom untuk model multimodal (Vision) Groq
    agar bisa dipakai di QueryProcessor layaknya LLM biasa.
    """

    def __init__(self, api_key: str, model: str = "llama-3.2-11b-vision-preview"):
        self.client = Groq(api_key=api_key)
        self.model = model

    def complete(self, prompt: str, image_url: str | None = None):
        """Kirim prompt + (opsional) image_url ke Groq Vision model."""
        content = [{"type": "text", "text": prompt}]
        if image_url:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
        )

        return response.choices[0].message.content.strip()
