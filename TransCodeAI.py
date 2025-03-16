import streamlit as st
import torch
import pickle
import math
import torch.nn as nn
import torch.nn.functional as F
import os

class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = self.word2idx
        self.itos = self.idx2word

    def __len__(self):
        return len(self.word2idx)

    @classmethod
    def tokenize(cls, text):
        return text.split()

    def tokenize_to_ids(self, tokens):
        return [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokens]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_k)
        return self.out(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, mask))
        x = self.norm2(x + self.ff(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
        x = self.norm2(x + self.cross_attn(x, enc_out, enc_out, src_mask))
        x = self.norm3(x + self.ff(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(num_layers)])

    def forward(self, src, src_mask=None):
        x = self.embedding(src)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        x = self.embedding(tgt)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.fc_out(x)

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, n_heads=4, d_ff=512, num_layers=2, max_len=5000):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab_size, d_model, n_heads, d_ff, num_layers, max_len)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, n_heads, d_ff, num_layers, max_len)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src, src_mask)
        return self.decoder(tgt, enc_out, src_mask, tgt_mask)

# Load model and vocabularies with caching
@st.cache_resource
def load_models():
    try:
        # Use relative paths for cloud compatibility
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Model checkpoints for both directions
        pseudo_to_code_checkpoint = os.path.join(current_dir, "pseudo2c++_transformer_seq2seq.pt")
        code_to_pseudo_checkpoint = os.path.join(current_dir, "c++2pseudo_transformer_seq2seq.pt")
        
        # Vocabulary files for both directions
        pseudo_vocab_file = os.path.join(current_dir, "pseudo_vocab.pkl")
        code_vocab_file = os.path.join(current_dir, "c++_vocab.pkl")

        # Check if all required files exist
        required_files = [
            pseudo_to_code_checkpoint,
            code_to_pseudo_checkpoint,
            pseudo_vocab_file,
            code_vocab_file
        ]
        
        if not all(os.path.exists(f) for f in required_files):
            st.error("Required model files are missing. Please ensure all files are present in the repository.")
            return None, None, None, None, None

        # Load vocabularies
        with open(pseudo_vocab_file, "rb") as f:
            pseudo_vocab = pickle.load(f)
        with open(code_vocab_file, "rb") as f:
            code_vocab = pickle.load(f)

        # Initialize models for both directions
        pseudo_to_code_model = TransformerSeq2Seq(
            src_vocab_size=len(pseudo_vocab),
            tgt_vocab_size=len(code_vocab),
            d_model=256,
            n_heads=4,
            d_ff=512,
            num_layers=2
        )
        
        code_to_pseudo_model = TransformerSeq2Seq(
            src_vocab_size=len(code_vocab),
            tgt_vocab_size=len(pseudo_vocab),
            d_model=256,
            n_heads=4,
            d_ff=512,
            num_layers=2
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model states
        pseudo_to_code_model.load_state_dict(torch.load(pseudo_to_code_checkpoint, map_location=device))
        code_to_pseudo_model.load_state_dict(torch.load(code_to_pseudo_checkpoint, map_location=device))
        
        # Move models to device and set to eval mode
        pseudo_to_code_model.to(device)
        code_to_pseudo_model.to(device)
        pseudo_to_code_model.eval()
        code_to_pseudo_model.eval()

        return pseudo_to_code_model, code_to_pseudo_model, pseudo_vocab, code_vocab, device
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

def convert_code(input_text, model, src_vocab, tgt_vocab, device, max_length=128):
    try:
        # Tokenize input
        input_tokens = input_text.split()
        src_ids = [src_vocab.word2idx.get(token, src_vocab.word2idx["<unk>"]) for token in input_tokens]
        
        # Generate conversion
        model.eval()
        with torch.no_grad():
            src = torch.tensor([src_ids], dtype=torch.long, device=device)
            src_mask = (src != src_vocab.word2idx["<pad>"]).unsqueeze(1).unsqueeze(2)
            enc_out = model.encoder(src, src_mask)

            ys = torch.tensor([[tgt_vocab.word2idx["<sos>"]]], dtype=torch.long, device=device)

            for _ in range(max_length - 1):
                tgt_mask = (ys != tgt_vocab.word2idx["<pad>"]).unsqueeze(1).unsqueeze(2)
                seq_len = ys.size(1)
                subsequent_mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool()
                tgt_mask = tgt_mask & ~subsequent_mask

                out = model.decoder(ys, enc_out, src_mask, tgt_mask)
                prob = out[:, -1, :]
                next_token = torch.argmax(prob, dim=1).item()

                ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)
                if next_token == tgt_vocab.word2idx["<eos>"]:
                    break

            output_ids = ys[0].cpu().numpy().tolist()
            output_tokens = [tgt_vocab.idx2word[i] for i in output_ids if i not in {tgt_vocab.word2idx["<sos>"], tgt_vocab.word2idx["<eos>"]}]
            
            return " ".join(output_tokens)
    except Exception as e:
        st.error(f"Conversion error: {str(e)}")
        return None

# Streamlit UI
st.set_page_config(
    page_title="TransCodeAI",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stTextArea {
        font-size: 16px;
        font-family: 'Courier New', Courier, monospace;
    }
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .stTab {
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("💻 Code Converter Settings")
    st.markdown("---")
    st.markdown("""
    ### About
    This tool converts between:
    - Pseudo Code ↔ Programming Code
    
    ### Tips
    - Use clear and structured input
    - Follow standard conventions
    - Keep code segments concise
    - Ensure proper indentation
    """)

# Main content
st.title("💻 TransCodeAI")

# Load models
pseudo_to_code_model, code_to_pseudo_model, pseudo_vocab, code_vocab, device = load_models()

if None in (pseudo_to_code_model, code_to_pseudo_model, pseudo_vocab, code_vocab, device):
    st.error("Failed to load the models. Please ensure all required files are present.")
    st.stop()

# Create tabs for different conversion directions
tab1, tab2 = st.tabs(["Pseudo Code → Code", "Code → Pseudo Code"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pseudo Code Input")
        pseudo_code = st.text_area(
            "Enter Pseudo Code",
            placeholder="BEGIN\n    READ number\n    IF number > 0 THEN\n        PRINT 'Positive'\n    ENDIF\nEND",
            height=300
        )
        
        if st.button("Convert to Code", type="primary", key="pseudo_to_code"):
            with st.spinner("Converting..."):
                generated_code = convert_code(pseudo_code, pseudo_to_code_model, pseudo_vocab, code_vocab, device)
                with col2:
                    st.subheader("Generated Code")
                    st.text_area("Code Output", generated_code if generated_code else "", height=300)

with tab2:
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Code Input")
        program_code = st.text_area(
            "Enter Code",
            placeholder="def check_number(num):\n    if num > 0:\n        print('Positive')",
            height=300
        )
        
        if st.button("Convert to Pseudo Code", type="primary", key="code_to_pseudo"):
            with st.spinner("Converting..."):
                generated_pseudo = convert_code(program_code, code_to_pseudo_model, code_vocab, pseudo_vocab, device)
                with col4:
                    st.subheader("Generated Pseudo Code")
                    st.text_area("Pseudo Code Output", generated_pseudo if generated_pseudo else "", height=300)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Dev: Soban & Ali</p>
    </div>
""", unsafe_allow_html=True)
