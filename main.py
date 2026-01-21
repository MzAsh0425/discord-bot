import os
import discord
from discord.ext import commands
from dotenv import load_dotenv

# --- RAG・Gemini関連のライブラリ ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# --- 1. 設定と準備 ---
load_dotenv() # .envファイルからAPIキーを読み込む
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Botが反応するチャンネル名（※自分のサーバーのチャンネル名に合わせて変更可能）
TARGET_CHANNEL_NAME = "stagea03-質問部屋"

# --- 2. 知識ベース（RAG）の構築 ---
def create_rag_chain():
    print("📂 ドキュメントを読み込んでいます...")
    
    # dataフォルダ内のすべての.txtファイルを読み込む
    loader = DirectoryLoader(
        './data', 
        glob="**/*.txt", 
        loader_cls=TextLoader,
        show_progress=True
    )
    documents = loader.load()
    
    if not documents:
        print("⚠️ 注意: dataフォルダにテキストファイルが見つかりません。テスト用のファイルがあるか確認してください。")
        return None

    print(f"✅ {len(documents)} 件のファイルを読み込みました。")

    # テキストを適切なサイズに分割（Geminiが理解しやすくするため）
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # ベクトル化（埋め込み）モデルの準備：GeminiのEmbeddingモデルを使用
    print("🧠 ベクトルデータベースを構築中...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # ベクトル検索エンジン(FAISS)に格納
    vector_store = FAISS.from_documents(texts, embeddings)
    
    # 回答生成モデルの準備：Gemini 1.5 Flash（高速・高性能）
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    # 検索と回答を繋ぐチェーンを作成
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}) # 上位3つの関連情報を参照
    )
    print("🚀 準備完了！Botを起動します。")
    return qa_chain

# 起動時にRAGチェーンを作成
qa_chain = create_rag_chain()

# --- 3. Discord Botのイベント設定 ---
intents = discord.Intents.default()
intents.message_content = True # メッセージの中身を読む権限
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'🤖 Logged in as {bot.user}')

@bot.event
async def on_message(message):
    # 自分自身のメッセージには反応しない
    if message.author == bot.user:
        return

    # 指定したチャンネル以外では反応しない
    if message.channel.name != TARGET_CHANNEL_NAME:
        return
    # メンション (@bot) されていなければ無視する
    if bot.user not in message.mentions:
            return
    # qa_chainが正しく作られていない場合は無視
    if qa_chain is None:
        return

    # ユーザーへの「考え中...」の表示
    async with message.channel.typing():
        try:
            # Geminiに質問を投げて回答を取得
            response = qa_chain.invoke(message.content)
            answer = response['result']
            
            # Discordに送信
            await message.channel.send(answer)
            
        except Exception as e:
            await message.channel.send(f"エラーが発生しました: {e}")
            print(f"Error: {e}")

# --- 4. Botの実行 ---
if DISCORD_TOKEN:
    bot.run(DISCORD_TOKEN)
else:
    print("❌ エラー: .envファイルに DISCORD_TOKEN が設定されていません。")