import torch
import source.stable_diffusion as stable_diffusion
import rp

def _get_stable_diffusion_singleton():
    import source.stable_diffusion as sd
    if sd._stable_diffusion_singleton is None:
        assert False, 'Please create a stable_diffusion.StableDiffusion instance before creating a label'
    return stable_diffusion._stable_diffusion_singleton

class BaseLabel:
    def __init__(self, name:str, embedding:torch.Tensor, device=None):
        #Later on we might have more sophisticated embeddings, such as averaging multiple prompts
        #We also might have associated colors for visualization, or relations between labels
        
        if device is None:
            device=_get_stable_diffusion_singleton().device

        self.name=name
        self.embedding=embedding.to(device)
        
    def get_sample_image(self):
        s = _get_stable_diffusion_singleton()
        with torch.no_grad():
            output=s.embeddings_to_imgs(self.embedding)[0]
        assert rp.is_image(output)
        return output
            
    def __repr__(self):
        return '%s(name=%s)'%(type(self).__name__,self.name)
        
class SimpleLabel(BaseLabel):
    def __init__(self, name:str, device=None):
        s = _get_stable_diffusion_singleton()
        super().__init__(name, s.get_text_embeddings(name).to(device), device=device)

class NegativeLabel(BaseLabel):
    def __init__(self, name:str, negative_prompt='', device=None):
        s = _get_stable_diffusion_singleton()
        
        if '---' in name:
            #You can use '---' in a prompt to specify the negative part
            name,additional_negative_prompt=name.split('---',maxsplit=1)
            negative_prompt+=' '+additional_negative_prompt
            
        self.negative_prompt=negative_prompt
        old_uncond_text=s.uncond_text
        try:
            s.uncond_text=negative_prompt
            embedding = s.get_text_embeddings(name)
            super().__init__(name, embedding, device=device)
        finally:
            s.uncond_text=old_uncond_text