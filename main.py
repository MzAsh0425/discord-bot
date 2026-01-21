import os
import discord
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from keep_alive import keep_alive # サーバー常時稼働用

# 環境変数の読み込み
load_dotenv()

# 設定
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TARGET_CHANNEL_NAME = "stagea03-質問部屋" # Botが反応するチャンネル名
DATA_DIR = "data" # テキストファイルを置くフォルダ

# Discordクライアントの設定
intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)

# グローバル変数としてチェーンを保持
qa_chain = None

def create_rag_chain():
    """RAGのチェーンを作成する関数（Gemini 2.0対応版）"""
    global qa_chain
    
    if not os.path.exists(DATA_DIR):
        print(f"フォルダ {DATA_DIR} が見つかりません。作成します。")
        os.makedirs(DATA_DIR)
        return None

    print("📂 ドキュメントを読み込んでいます...")
    try:
        # テキストファイルを読み込む (show_progress=Trueで進捗表示)
        loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
        documents = loader.load()
        
        if not documents:
            print("⚠️ テキストファイルが見つかりませんでした。")
            return None

        print(f"✅ {len(documents)} 件のファイルを読み込みました。")

        # テキストを分割する
        # Gemini 2.0のAPI制限を回避するため、チャンクサイズを大きめにしてリクエスト回数を減らす
        text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        print("🧠 ベクトルデータベースを構築中...")
        # Embeddingsモデルの設定
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # ベクトルDB作成
        db = FAISS.from_documents(texts, embeddings)
        
        # Retrieverの設定
        retriever = db.as_retriever()

        # LLM（Gemini 2.0 Flash）の設定
        # max_retries=10: エラーが出ても10回まで自動で待ちながら再試行する（重要）
        # transport="rest": 通信を安定させるための設定
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0,
            max_retries=10,
            transport="rest" 
        )

        # QAチェーンの作成
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        print("🚀 RAGチェーンの準備が完了しました。")
        return qa_chain

    except Exception as e:
        print(f"❌ 初期化中にエラーが発生しました: {e}")
        return None

@bot.event
async def on_ready():
    print(f'🚀 準備完了！Botを起動します。')
    print(f'ログインしました: {bot.user}')
    # 起動時にRAGチェーンを構築
    create_rag_chain()

@bot.event
async def on_message(message):
    # 自分自身のメッセージには反応しない
    if message.author == bot.user:
        return

    # 指定したチャンネル以外では反応しない
    if message.channel.name != TARGET_CHANNEL_NAME:
        return

    # Botへのメンションが含まれていない場合は無視
    if bot.user not in message.mentions:
        return

    # "考え中..." の表示
    async with message.channel.typing():
        try:
            # RAGチェーンがない場合は再構築を試みる
            if qa_chain is None:
                create_rag_chain()
                if qa_chain is None:
                    await message.channel.send("知識データの読み込みに失敗しているため、回答できません。")
                    return

            # メッセージからメンションを除去してクエリにする
            query = message.content.replace(f'<@&{bot.user.id}>', '').replace(f'<@{bot.user.id}>', '')

            # Geminiに質問を投げる
            # invokeを使うことで、ここでも自動リトライが効く
            response = await bot.loop.run_in_executor(None, qa_chain.invoke, query)
            answer = response['result']
            
            await message.channel.send(answer)
            
        except Exception as e:
            # それでもエラーが出た場合はログに出す
            error_msg = str(e)
            print(f"Error: {error_msg}")
            
            # 429エラー（制限）の場合はユーザーに分かりやすく伝える
            if "429" in error_msg:
                await message.channel.send("申し訳ありません。Gemini 2.0の利用制限（アクセス集中）のため、少し時間を置いてからもう一度話しかけてください。")
            else:
                await message.channel.send(f"エラーが発生しました: {e}")

# Webサーバーを裏で動かす（24時間稼働用）
keep_alive()

# Botの実行
if DISCORD_TOKEN:
    bot.run(DISCORD_TOKEN)