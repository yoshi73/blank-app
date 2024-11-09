pip install streamlit oci openai

import streamlit as st
import time
import json
from oci.config import from_file
from oci.object_storage import ObjectStorageClient
from oci.ai_speech import AIServiceSpeechClient
from oci.ai_speech.models import CreateTranscriptionJobDetails, ObjectListInlineInputLocation, ObjectLocation, OutputLocation, TranscriptionModelDetails
from datetime import datetime
from openai import OpenAI

config = from_file()
object_storage = ObjectStorageClient(config)
namespace = object_storage.get_namespace().data
st.title('Whisper demo🎤')
bucket_name = "bucket-20241109-OCISpeech" #バケット名を入れます
uploaded_file = st.file_uploader(
    "音声/動画ファイルをドラッグ＆ドロップまたは選択してください", type=['mp3', 'wav', 'mp4', 'm4a', 'aac'])
if uploaded_file is not None:
    file_content = uploaded_file.getvalue()
    object_name = uploaded_file.name
    object_storage.put_object(namespace_name=namespace, bucket_name=bucket_name,
                              object_name=object_name, put_object_body=file_content)
    st.success(f"ファイル '{object_name}' がバケット '{bucket_name}' にアップロードされました！")
    st.session_state.uploaded_file_name = object_name

    st.write("OCI Speechの処理を開始します。")
    speech_client = AIServiceSpeechClient(config)
    audio_file_uri = f'oci://{bucket_name}@{namespace}/{
        st.session_state.uploaded_file_name}'

    create_transcription_job_details = CreateTranscriptionJobDetails(
        compartment_id='ocid1.tenancy.oc1..aaaaaaaaruzw4uj5pbj5orefhh43yjnlrtem7kxtvb3fpbjraxxrvpc4atiq',    # コンパートメントIDを入れます
        display_name="PythonSDKSampleTranscriptionJob",
        description="Transcription job created by Python SDK",
        input_location=ObjectListInlineInputLocation(
            location_type="OBJECT_LIST_INLINE_INPUT_LOCATION",
            object_locations=[ObjectLocation(
                namespace_name=namespace,
                bucket_name=bucket_name,
                object_names=[st.session_state.uploaded_file_name])]),
        output_location=OutputLocation(
            namespace_name=namespace, bucket_name=bucket_name),
        model_details=TranscriptionModelDetails(
            model_type="WHISPER_MEDIUM", language_code="ja")
    )

    response = speech_client.create_transcription_job(
        create_transcription_job_details)
    job_id = response.data.id

    status_placeholder = st.empty()
    status_message = ""
    start_time = datetime.now()
    with st.spinner('処理中です。しばらくお待ちください...'):
        job_finished = False
        while not job_finished:
            current_time = datetime.now()
            processing_time = current_time - start_time
            job_details = speech_client.get_transcription_job(job_id).data
            job_status = job_details.lifecycle_state
            processing_time_str = str(processing_time).split('.')[0]
            new_status_message = f"Job Status: {
                job_status} - Processing Time: {processing_time_str}"
            if new_status_message != status_message:
                status_placeholder.write("更新中...")
                time.sleep(1)
                status_placeholder.empty()
            status_placeholder.write(new_status_message)
            status_message = new_status_message
            st.session_state.job_id = job_id
            if job_status in ["SUCCEEDED"]:
                job_finished = True
                output_location = job_details.output_location
                bucket_name = output_location.bucket_name
                namespace_name = output_location.namespace_name
                prefix = output_location.prefix
                object_storage_client = ObjectStorageClient(config)
                list_objects_response = object_storage_client.list_objects(
                    namespace_name=namespace_name,
                    bucket_name=bucket_name,
                    prefix=prefix,
                    fields="name"
                )
                file_name = list_objects_response.data.objects[0].name
                get_object_response = object_storage_client.get_object(
                    namespace_name=namespace_name,
                    bucket_name=bucket_name,
                    object_name=file_name
                )
                file_content = get_object_response.data.text
                transcriptions_json = json.loads(file_content)
                for transcription in transcriptions_json.get('transcriptions', []):
                    # LLMで整形処理を実施します。
                    client = OpenAI(
                        api_key="sk-***", # apiキーを取り込み
                    )
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": "以下に内容について句読点、改行(<br>)を入れてください。また最後に要約した内容を別途追記してください。：\n"
                                + transcription.get('transcription'),
                            }
                        ],
                        model="gpt-3.5-turbo",
                    )
                    st.markdown(
                        chat_completion.choices[0].message.content, unsafe_allow_html=True)
            else:
                time.sleep(5)
else:
    st.write("音声/動画ファイルをアップロードして、処理を開始してください。")
