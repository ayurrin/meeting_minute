import os
import streamlit as st
from pydub import AudioSegment
import speech_recognition as sr
import whisper
import openai
import tempfile
import moviepy.editor as mp

# GPT-3.5 Turboモデルの認証
openai.api_key = "YOUR_API_KEY"


# 音声ファイルの変換
def convert_to_wav(input_file, output_file):
    file_extension = os.path.splitext(input_file)[1]

    if file_extension == ".mp3":
        audio = AudioSegment.from_mp3(input_file)
        audio.export(output_file, format="wav")
        return output_file

    if file_extension == ".mp4":
        video = mp.VideoFileClip(input_file)
        audio = video.audio
        audio.write_audiofile(output_file, codec="pcm_s16le")
        return output_file

    raise ValueError("Unsupported file format: {}".format(file_extension))


# 音声認識（Whisper）
def transcribe_with_whisper(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    st.write(result)
    return result["text"]


# 音声認識（SpeechRecognition）
def transcribe_with_speechrecognition(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    return r.recognize_google(audio, language="ja-JP")


# 音声ファイルの変換と音声認識の処理
def process_transcription(input_file, output_file, engine):
    # try:
        converted_file = convert_to_wav(input_file, output_file)
        st.write("ファイルが変換されました: {}".format(converted_file))

        if engine == "whisper":
            transcription = transcribe_with_whisper(converted_file)
            st.write("音声認識結果 (Whisper): {}".format(transcription))
        elif engine == "speechrecognition":
            transcription = transcribe_with_speechrecognition(converted_file)
            st.write("音声認識結果 (SpeechRecognition): {}".format(transcription))
        else:
            raise ValueError("Invalid engine: {}".format(engine))
    # except ValueError as e:
    #     st.write(str(e))
    # except FileNotFoundError:
    #     st.write("ファイルが見つかりません。")
    

def set_api():
    openai.api_key = st.session_state["api_key"]

prompt = """
会議の議事録を作成します。以下の制約条件に従って、会話の内容から要点を抽出し、明確で意味がわかりやすい文章にまとめてください。

制約条件:
- 会話の内容に基づいてまとめること
- 発言の内容から言いたいことを抽出すること
- 抽出した内容から関連したものを総合して一つの要点とすること
- 各要点は100字程度でまとめること
- Markdownでまとめること

最後にサマリーとして以下の事項を出力してください
- 会議の目的:
- 決定事項:
- 今後の課題:
- Todo:
"""
# Streamlitアプリの作成
st.markdown("## 議事録要約アプリ")

#API入力
api_key = st.text_input("OpenAI APIキーを入力してください", on_change=set_api, key='api_key')
# 選択したエンジン
engine = st.selectbox("音声認識エンジンの選択", ["whisper", "speechrecognition"])

# 音声ファイルのアップロード
uploaded_file = st.file_uploader("音声ファイルをアップロードしてください", type=["mp3", "mp4"])
if uploaded_file is not None:
    if st.button("開始"):


        # 一時ディレクトリの作成
        with tempfile.TemporaryDirectory() as temp_dir:
            # アップロードされたファイルの保存
            input_file_path = os.path.join(temp_dir, "input_file" + os.path.splitext(uploaded_file.name)[1])
            with open(input_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 出力ファイルのパス
            output_file_path = os.path.join(temp_dir, "output.wav")

            
            transcription = process_transcription(input_file_path, output_file_path, engine)
            
            # 要約結果の表示
            if st.session_state["api_key"] == "":
                import pickle

                # 保存したresponseオブジェクトを読み込むファイルパス
                file_path = "response.pkl"

                # pickleファイルからresponseオブジェクトを読み込む
                with open(file_path, "rb") as file:
                    response = pickle.load(file)
            else:
                response = openai.Completion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": transcription},
                    ],
                )
            summary = response.choices[0].message.content

            st.markdown("### 議事録要約結果")
            st.markdown(summary)
