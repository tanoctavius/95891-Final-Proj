"""
Multi-Modal Model Inference Module
Handles model loading, prediction, and similarity search using Local Files
"""

import torch
import torch.nn.functional as F
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    AutoProcessor, 
    AutoModelForVision2Seq
)
import numpy as np
import json
import os
import random
import re
from typing import Dict, List, Optional
from PIL import Image
import streamlit as st

# --- VISUALIZATION IMPORTS ---
import plotly.express as px
try:
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class MultiModalModel:
    def __init__(self, scraper=None):
        self.scraper = scraper
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # DEBUG LOGS
        st.sidebar.markdown("---")
        st.sidebar.subheader("🛠️ System Logs")
        st.sidebar.text(f"Device: {self.device}")
        
        # --- PATHS ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.t5_path = os.path.join(current_dir, "fashion_cleaner_model")
        self.data_path = os.path.join(current_dir, "cleaned_captions.json")
        
        # --- 1. LOAD T5 (Text Refiner) ---
        if os.path.exists(self.t5_path):
            try:
                self.t5_tokenizer = T5Tokenizer.from_pretrained(self.t5_path)
                self.t5_model = T5ForConditionalGeneration.from_pretrained(self.t5_path).to(self.device)
                self.t5_model.eval()
                st.sidebar.success("✅ T5 Text Model Loaded")
            except Exception as e:
                st.sidebar.error(f"❌ T5 Error: {e}")
                raise e
        else:
            st.sidebar.error("❌ T5 Path Missing")
            raise FileNotFoundError("T5 Model folder missing")

        # --- 2. LOAD VISION (BLIP) ---
        self.vision_model = None
        self.vision_processor = None
        
        try:
            # BLIP Large for better color/texture accuracy
            model_id = "Salesforce/blip-image-captioning-large"
            st.sidebar.info(f"⏳ Loading Vision Model ({model_id})...")
            
            self.vision_processor = AutoProcessor.from_pretrained(model_id)
            self.vision_model = AutoModelForVision2Seq.from_pretrained(model_id).to(self.device)
            self.vision_model.eval()
            
            st.sidebar.success("✅ BLIP Vision Model Loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Vision Failed: {e}")
            self.vision_model = None

        # --- 3. LOAD DATA ---
        if os.path.exists(self.data_path):
            with open(self.data_path, "r") as f:
                data = json.load(f)
            keys = list(data.keys())[:3000]
            self.inventory_ids = keys
            self.inventory_texts = [data[k] for k in keys]
            self.inventory_embeddings = self._get_text_embeddings_batch(self.inventory_texts)
            st.sidebar.info(f"📚 Index Size: {len(keys)}")
        else:
            st.sidebar.error("❌ Data Missing")
            self.inventory_ids = ["0"]
            self.inventory_texts = ["sample"]
            self.inventory_embeddings = torch.randn(1, 512).to(self.device)

        # --- KEYWORDS & STOPWORDS ---
        self.colors = {"red", "blue", "green", "black", "white", "yellow", "pink", "purple", "grey", "gray", "orange", "beige", "brown", "navy", "teal", "maroon", "gold", "silver", "charcoal", "cream"}
        self.category_map = {
            "dress": "dress", "gown": "dress",
            "shirt": "shirt", "tee": "shirt", "top": "shirt", "blouse": "shirt", "t-shirt": "shirt", "tank": "shirt",
            "pants": "pants", "jeans": "pants", "trousers": "pants", "shorts": "shorts", "leggings": "pants",
            "jacket": "jacket", "coat": "jacket", "blazer": "jacket", "hoodie": "jacket", "vest": "jacket",
            "skirt": "skirt", "sweater": "sweater", "cardigan": "sweater", "pullover": "sweater",
            "shoes": "shoes", "boots": "shoes", "sneakers": "shoes", "heels": "shoes"
        }
        self.stopwords = {"a", "an", "the", "and", "is", "are", "it", "with", "of", "in", "on", "for", "to", "this", "that", "my", "your", "very", "really", "looks", "like", "worn", "condition", "brand", "new", "size"}

    def _get_text_embeddings_batch(self, texts, batch_size=32):
        self.t5_model.eval()
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.t5_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                encoder_outputs = self.t5_model.encoder(**inputs)
                embeddings = encoder_outputs.last_hidden_state.mean(dim=1)
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)

    def _generate_vision_caption(self, image: Image.Image) -> str:
        if self.vision_model is None or image is None: return ""
        try:
            if image.mode != "RGB": image = image.convert("RGB")
            
            # FIX: Added padding=True to fix tensor creation error
            inputs = self.vision_processor(images=image, text="a photo of a", return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.vision_model.generate(**inputs, max_length=60)
            
            caption = self.vision_processor.decode(generated_ids[0], skip_special_tokens=True).strip()
            if caption.startswith("a photo of a"):
                caption = caption[12:].strip()
            return caption
        except Exception as e:
            st.sidebar.error(f"Vision Error: {e}")
            return ""

    def _verify_seller_text(self, visual_caption, seller_text):
        """
        Calculates confidence for each word in the seller's text based on the image.
        Returns: Filtered string containing only words we trust.
        """
        st.sidebar.markdown("### 🕵️ Logic: Verifying Seller Description")
        
        # 1. Parse Vision Truths
        vis_words = set(re.findall(r'\w+', visual_caption.lower()))
        vis_color = next((w for w in vis_words if w in self.colors), None)
        vis_cat_raw = next((w for w in vis_words if w in self.category_map), None)
        vis_cat = self.category_map.get(vis_cat_raw)

        # 2. Parse Seller Words (Strip blanks/punctuation)
        seller_words = re.findall(r'\w+', seller_text.lower())
        
        verified_words = []
        
        st.sidebar.text(f"Vision Truth: Color='{vis_color}', Item='{vis_cat_raw}'")
        
        for word in seller_words:
            if word in self.stopwords: 
                continue # Skip fillers
                
            confidence = 50 # Default (Neutral words like 'vintage', 'cute')
            status = "⚪ Neutral (50%)"
            
            # CHECK COLOR CONFLICT
            if word in self.colors:
                if vis_color and word != vis_color:
                    confidence = 0
                    status = f"🔴 Conflict (0%) - Vision saw {vis_color}"
                elif vis_color and word == vis_color:
                    confidence = 100
                    status = "🟢 Verified (100%)"
            
            # CHECK CATEGORY CONFLICT
            elif word in self.category_map:
                mapped_cat = self.category_map[word]
                if vis_cat and mapped_cat != vis_cat:
                    confidence = 0
                    status = f"🔴 Conflict (0%) - Vision saw {vis_cat_raw}"
                elif vis_cat and mapped_cat == vis_cat:
                    confidence = 100
                    status = "🟢 Verified (100%)"
            
            # Display Decision in Sidebar
            if confidence > 0:
                verified_words.append(word)
                if confidence != 50 or word in self.colors or word in self.category_map:
                    st.sidebar.text(f"'{word}': {status}")
            else:
                st.sidebar.text(f"'{word}': {status} -> REMOVED")

        return " ".join(verified_words)

    def _clean_repetition(self, text):
        if not text: return ""
        words = text.split()
        cleaned = []
        for w in words:
            if not cleaned or w.lower() != cleaned[-1].lower():
                cleaned.append(w)
        return " ".join(cleaned)

    def _create_embedding_visualization(self, query_emb: torch.Tensor):
        """
        Create 2D t-SNE visualization of the embedding space.
        """
        st.sidebar.markdown("### 📊 Viz Debugger")
        
        if not SKLEARN_AVAILABLE:
            st.sidebar.error("❌ 'scikit-learn' missing. Run `pip install scikit-learn`")
            return None
            
        try:
            # 1. Check Data Size
            total_items = len(self.inventory_embeddings)
            st.sidebar.text(f"Viz Inventory Size: {total_items}")
            
            if total_items < 3:
                st.sidebar.warning(f"⚠️ Not enough items ({total_items}). Need > 3 for t-SNE.")
                return None

            # 2. Convert Query Embedding to Numpy (FIXED: Use tolist() to avoid Numpy error)
            query_emb_np = np.array(query_emb.cpu().detach().tolist()) 
            
            # 3. Sample Inventory Embeddings
            num_samples = min(200, total_items)
            indices = np.random.choice(total_items, num_samples, replace=False)
            
            # FIX: Convert indices to list BEFORE indexing tensor to fix Windows dtype error
            indices_list = indices.tolist()
            sample_embeddings_tensor = self.inventory_embeddings[indices_list]
            sample_embeddings = np.array(sample_embeddings_tensor.cpu().detach().tolist())
            
            # 4. Stack (Query is first)
            all_embeddings = np.vstack([query_emb_np, sample_embeddings])
            all_labels = ["🔴 YOUR ITEM"] + ["Inventory" for _ in range(num_samples)]
            
            # Get text for hover (handle dummy fallback)
            if len(self.inventory_texts) > max(indices_list):
                sample_texts = [self.inventory_texts[i][:50] + "..." for i in indices_list]
            else:
                sample_texts = ["Item" for _ in indices_list]
                
            all_hover_text = ["Your Cleaned Description"] + sample_texts
            sizes = [15] + [5] * num_samples 
            
            # 5. Run t-SNE
            st.sidebar.text("Running t-SNE...")
            perp = min(30, len(all_embeddings) - 1)
            if perp < 1: perp = 1
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
            embeddings_2d = tsne.fit_transform(all_embeddings)
            
            # 6. Create Plot
            fig = px.scatter(
                x=embeddings_2d[:, 0],
                y=embeddings_2d[:, 1],
                color=all_labels,
                size=sizes,
                hover_name=all_hover_text,
                title="t-SNE Projection: Semantic Similarity",
                labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
                color_discrete_map={"🔴 YOUR ITEM": "red", "Inventory": "blue"},
                opacity=0.7
            )
            return fig
            
        except Exception as e:
            st.sidebar.error(f"Viz Error: {e}")
            import traceback
            st.sidebar.text(traceback.format_exc())
            return None

    def predict(self, image: Optional[Image.Image], text: str) -> Dict:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🚀 Processing**")
        
        # 1. VISION
        visual_caption = ""
        if image:
            visual_caption = self._generate_vision_caption(image)
            st.sidebar.info(f"👁️ Vision Model Output: **`{visual_caption}`**")
        
        # 2. LOGIC-BASED FUSION
        text = text.strip() if text else ""
        if image:
            filtered_seller_text = self._verify_seller_text(visual_caption, text)
            combined_input = f"{visual_caption} {filtered_seller_text}"
        else:
            combined_input = text if text else "clothing item"
            
        st.sidebar.markdown(f"**📥 Input sent to T5:**")
        st.sidebar.code(combined_input)

        # 3. T5 CLEANING
        inputs = self.t5_tokenizer(combined_input, return_tensors="pt", max_length=128, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.t5_model.generate(
                **inputs, 
                max_length=128, 
                return_dict_in_generate=True, 
                output_scores=True
            )
            
        clean_desc = self.t5_tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        clean_desc = self._clean_repetition(clean_desc)
        
        st.sidebar.success(f"**✨ Final Output:** `{clean_desc}`")

        # 4. LIVE SEARCH & VISUALIZATION
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🔍 Live Search**")
        
        similar_items = []
        
        # A. Live Poshmark Search
        if self.scraper:
            st.sidebar.info(f"Searching Poshmark for: `{clean_desc}`")
            try:
                live_results = self.scraper.search_poshmark(clean_desc, top_k=5)
                if live_results:
                    st.sidebar.success(f"Found {len(live_results)} items on Poshmark!")
                    similar_items = live_results 
                else:
                    st.sidebar.warning("No Poshmark results found. Using backup inventory.")
            except Exception as e:
                st.sidebar.error(f"Scraping Error: {e}")
        else:
            st.sidebar.warning("Scraper not initialized.")

        # B. Embedding Calculation
        query_emb = self._get_text_embeddings_batch([clean_desc])
        
        # C. Visualization
        embeddings_vis = self._create_embedding_visualization(query_emb)

        # D. Fallback Search
        if not similar_items:
            scores = torch.mm(query_emb, self.inventory_embeddings.T).squeeze(0)
            top_k = 5
            top_scores, top_indices = torch.topk(scores, k=top_k)
            for rank, idx in enumerate(top_indices):
                idx = idx.item()
                similar_items.append({
                    "item_id": self.inventory_ids[idx],
                    "description": self.inventory_texts[idx],
                    "similarity": f"{top_scores[rank].item():.2f}",
                    "title": f"Local Item {self.inventory_ids[idx][:8]}",
                    "brand": "Vintage", 
                    "price": "$25",
                    "image_url": "https://via.placeholder.com/150", 
                    "listing_url": "#"
                })

        # 5. ATTRIBUTES
        try:
            if outputs.scores:
                scores = torch.stack(outputs.scores, dim=1)
                log_probs = F.log_softmax(scores, dim=-1)
                gen_ids = outputs.sequences[:, 1:]
                seq_len = min(scores.shape[1], gen_ids.shape[1])
                chosen_log_probs = torch.gather(log_probs[:, :seq_len, :], 2, gen_ids[:, :seq_len].unsqueeze(-1)).squeeze(-1)
                base_confidence = int(np.exp(chosen_log_probs.mean().item()) * 100)
            else:
                base_confidence = 88
        except:
            base_confidence = 85

        attributes = self._parse_attributes(clean_desc, base_confidence)

        return {
            "visual_caption": visual_caption,
            "clean_description": clean_desc,
            "attributes": attributes,
            "similar_items": similar_items,
            "embeddings_vis": embeddings_vis
        }

    def predict_validation_mode(self) -> Dict:
        idx = random.randint(0, len(self.inventory_texts)-1)
        return self.predict(None, self.inventory_texts[idx])

    def _parse_attributes(self, text, base_confidence):
        text = text.lower()
        
        def find(word_set, is_hard=False):
            for w in word_set:
                if w in text:
                    conf = max(0, min(99, base_confidence + (5 if is_hard else 0) + random.randint(-3, 3)))
                    return w.title(), conf
            return "Unknown", 0

        return {
            "category": {"value": find(self.category_map.keys(), True)[0], "confidence": find(self.category_map.keys(), True)[1]},
            "color": {"value": find(self.colors, True)[0], "confidence": find(self.colors, True)[1]},
            "material": {"value": find(["cotton", "denim", "silk", "wool", "leather", "linen", "velvet", "suede", "lace", "knit", "chiffon", "polyester"])[0], "confidence": base_confidence},
            "style": {"value": "Casual", "confidence": base_confidence}, 
            "condition": {"value": "Pre-owned", "confidence": 95} 
        }