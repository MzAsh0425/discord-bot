import os
import discord
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    """RAGのチェーンを作成する関数（精度向上・Gemini 2.0対応版）"""
    global qa_chain
    
    if not os.path.exists(DATA_DIR):
        print(f"フォルダ {DATA_DIR} が見つかりません。作成します。")
        os.makedirs(DATA_DIR)
        return None

    print("📂 ドキュメントを読み込んでいます...")
    try:
        # テキストファイルを読み込む
        loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
        documents = loader.load()
        
        if not documents:
            print("⚠️ テキストファイルが見つかりませんでした。")
            return None

        print(f"✅ {len(documents)} 件のファイルを読み込みました。")

        # 【改善1】テキスト分割を「Recursive（再帰的）」に変更
        # 文章の意味の区切れ（改行や句読点）を優先して切るため、文脈が途切れにくい
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 1塊のサイズ（少し小さくして具体性を高める）
            chunk_overlap=200 # 前後の重複
        )
        texts = text_splitter.split_documents(documents)

        print("🧠 ベクトルデータベースを構築中...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.from_documents(texts, embeddings)
        
        # 【改善2】Retrieverの設定（検索件数を増やす）
        # k=6: 関連する文章をトップ6件まで引っ張ってくる（デフォルトは4）
        retriever = db.as_retriever(search_kwargs={"k": 6})

        # LLMの設定
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0, # 0にすることで、創造性を排除し事実に忠実にさせる
            max_retries=10,
            transport="rest" 
        )

        # 【改善3】カスタムプロンプト（AIへの詳細な指示書）を作成
        # ここで「具体的に答えろ」「ページ数は不要」と指示する
        template = """
        あなたは提供された資料に基づいて質問に答える専門のアシスタントです。
        以下の「参照ドキュメント」の内容のみを使用して、質問に答えてください。
        
        【重要なルール】
        1. 抽象的な要約ではなく、ドキュメントに書かれている「具体的な詳細、数値、手順」をそのまま引用して答えてください。
        2. 「〇〇ページに書いてあります」のようなページ情報の回答は不要です。そのページに書かれている中身を答えてください。
        3. ドキュメントに答えが書かれていない場合は、「提供された資料にはその情報が含まれていません」と正直に答えてください。嘘をつかないでください。
        
        参照ドキュメント:
        {context}

        質問: {question}

        回答:
        """
        
        PROMPT = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )

        # QAチェーンの作成
        # chain_type_kwargsを使ってプロンプトを渡す
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": PROMPT}
        )
        print("🚀 RAGチェーン（高精度版）の準備が完了しました。")
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